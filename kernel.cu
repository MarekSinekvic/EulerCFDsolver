#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLEW_STATIC
#include <GL/freeglut.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "Library.cu";
#include "lodepng.h";

using namespace std;
//using namespace Library;

constexpr auto PI = 3.14159265359f;

bool paused = false;
int frame = 0;

typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;

struct Cell {
public:
    Vector3f velocity;
    float pressure, heat = 0.f;
    Vector3f color;

    float solidness = 0.f;
    __device__ __host__ Cell(Vector3f velocity = Vector3f(0.f,0.f)) {
        this->velocity = velocity;
        this->pressure = 0.f;
        this->color = Vector3f(0.f,0.f,0.f);
    }
};

__constant__ float DT = .4f;

const int GridResolution = 128*1;
const int GridResolutionX = GridResolution;
const int GridResolutionY = GridResolution; 
Cell* h_grid, *d_grid;

int ResolutionScale = 1;

__constant__ float density = .1f;
__constant__ float viscosity = .0f;
__constant__ float vorticity = .0f;
__constant__ float heatExapnsion = .22f;

__constant__ float color_diffusion = 0.01f*0;
__constant__ float heat_diffusion = 0.01f;
__constant__ float heatOverhead = 5.f;

__constant__ float color_generation_intens = .015f;
__constant__ float velocity_generation_intens = 10.f;
__constant__ float pressure_generation_intens = 3.5f;

Vector3f mousePos = Vector3f(0.f,0.f);
Vector3f mouseMovement = Vector3f(0.f,0.f);
int mouseButton = -1;

const Vector3f fire_color = Vector3f(1, .46, 0.);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ int sign(float x) { if (signbit(x) == 0) return -1; else return 1; }
template<class T>
__device__ T lerpInGrid(Cell* dgrid, Vector3f pos, T Cell::* quantity) {
    int2 cell = make_int2((int)pos.x,(int)pos.y);

    Vector3f delta = Vector3f((pos.x - ((float)cell.x + .5f)), ((float)pos.y - (cell.y + .5f)));
    int2 direction = make_int2(sign(delta.x), sign(delta.y));

    if (cell.x + direction.x < 0 || cell.x + direction.x > GridResolutionX - 1) direction.x = 0;
    if (cell.y + direction.y < 0 || cell.y + direction.y > GridResolutionY - 1) direction.y = 0;

    Cell cells[4] = {
        dgrid[(cell.y) * GridResolutionX + (cell.x)],
        dgrid[(cell.y) * GridResolutionX + (cell.x + direction.x)],
        dgrid[(cell.y + direction.y) * GridResolutionX + (cell.x)],
        dgrid[(cell.y + direction.y) * GridResolutionX + (cell.x + direction.x)],
    };
    
    T l1 = lerp(cells[0].*quantity, cells[1].*quantity, abs(delta.x));
    T l2 = lerp(cells[2].*quantity, cells[3].*quantity, abs(delta.x));

    T l = lerp(l1, l2, abs(delta.y));

    if (cell.x == 0 || cell.x == GridResolutionX - 1) l = T();
    if (cell.y == 0 || cell.y == GridResolutionY - 1) l = T();

    return l;
}
__device__ bool IsOnBound(int i, int j) {

    if (i == 0) return true;
    if (i == GridResolutionY - 1) return true;
    if (j == 0) return true;
    if (j == GridResolutionX - 1) return true;

    return false;
}
/*__global__ void CalculateVortice(Cell* dgrid) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;

    if (dgrid[I].solidness > 0.f) return;

    int i = (int)((float)I / GridResolutionX);
    int j = I % GridResolutionX;

    float DX = 1.f / GridResolutionX;
    float DY = 1.f / GridResolutionY;

    int aj = I + 1, naj = I - 1;
    int ai = I + GridResolutionY, nai = I - GridResolutionX;

    float vortice = (dgrid[aj].velocity.y - dgrid[naj].velocity.y) - (dgrid[ai].velocity.x - dgrid[nai].velocity.x);
    dgrid[I].vortice = vortice;
}*/
__device__ void BoundValues(Cell* dgrid) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    float maxVel = 100.f;
    float maxPressure = 100.f;
    if (dgrid[I].pressure < -maxPressure) dgrid[I].pressure = -maxPressure;
    if (dgrid[I].pressure > maxPressure) dgrid[I].pressure = maxPressure;

    if (dgrid[I].velocity.x > maxVel) dgrid[I].velocity.x = maxVel;
    if (dgrid[I].velocity.x < -maxVel) dgrid[I].velocity.x = -maxVel;
    if (dgrid[I].velocity.y > maxVel) dgrid[I].velocity.y = maxVel;
    if (dgrid[I].velocity.y < -maxVel) dgrid[I].velocity.y = -maxVel;
}
__global__ void AdvectionKernel(Cell* dgrid) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;

    if (dgrid[I].solidness > 0.f) return;

    int i = (int)((float)I / GridResolutionX);
    int j = I % GridResolutionX;

    float DX = 1.f / GridResolutionX;
    float DY = 1.f / GridResolutionY;

    int aj = I + 1, naj = I - 1;
    int ai = I + GridResolutionY, nai = I - GridResolutionX;

    float ni = (((float)i + .5f) + (dgrid[I].velocity.y) * DT * DX);
    float nj = (((float)j + .5f) + (dgrid[I].velocity.x) * DT * DY);

    if (IsOnBound(i, j)) return;

    Vector3f startVelocity = dgrid[I].velocity;
    Vector3f newVelocity = dgrid[I].velocity;

    Vector3f advectVelocity = lerpInGrid(dgrid, Vector3f(nj, ni), &Cell::velocity);
    newVelocity = advectVelocity;

    float advectHeat = lerpInGrid(dgrid, Vector3f(nj, ni), &Cell::heat);
    dgrid[I].heat = advectHeat;

    Vector3f advectColor = lerpInGrid(dgrid, Vector3f(nj, ni), &Cell::color);
    Vector3f colorDer = Vector3f(
        (dgrid[aj].color.x + dgrid[naj].color.x + dgrid[ai].color.x + dgrid[nai].color.x) - 4.f * dgrid[I].color.x,
        (dgrid[aj].color.y + dgrid[naj].color.y + dgrid[ai].color.y + dgrid[nai].color.y) - 4.f * dgrid[I].color.y,
        (dgrid[aj].color.z + dgrid[naj].color.z + dgrid[ai].color.z + dgrid[nai].color.z) - 4.f * dgrid[I].color.z
    );
    dgrid[I].color = advectColor;
    dgrid[I].color += colorDer * color_diffusion * DT;

    /*float heatDer = dgrid[aj].heat + dgrid[naj].heat+ dgrid[ai].heat+ dgrid[nai].heat - 4.f* dgrid[I].heat;
    float advectHeat = lerpInGrid(dgrid, Vector3f(nj, ni), &Cell::heat);
    dgrid[I].heat = advectHeat;
    dgrid[I].heat += heatDer * heat_diffusion * DT;*/

    //dgrid[I].density = lerpInGrid(dgrid, Vector3f(nj, ni), &Cell::density);

    Vector3f visousLaplacian = ((dgrid[aj].velocity + dgrid[naj].velocity) + (dgrid[ai].velocity + dgrid[nai].velocity)) - 4.f*dgrid[I].velocity;
    newVelocity += visousLaplacian * DT * viscosity;

    //newVelocity += Vector3f(dgrid[ai].vortice - dgrid[nai].vortice, dgrid[naj].vortice- dgrid[aj].vortice) * vorticity * DT;

    dgrid[I].velocity = newVelocity;
    BoundValues(dgrid);
}
__global__ void PressureKernel(Cell* dgrid, int ItersCount = 1) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;

    if (dgrid[I].solidness > 0.f) return;

    int i = (int)((float)I / GridResolutionX);
    int j = I % GridResolutionX;

    float DX = 1.f / GridResolutionX;
    float DY = 1.f / GridResolutionY;

    int aj = I + 1, naj = I - 1;
    int ai = I + GridResolutionY, nai = I - GridResolutionX;
    if (IsOnBound(i, j)) return;

    float h = (dgrid[aj].heat + dgrid[naj].heat) + (dgrid[ai].heat + dgrid[nai].heat) - 4.f * dgrid[I].heat;
    float d = (dgrid[aj].velocity.x - dgrid[naj].velocity.x) + (dgrid[ai].velocity.y - dgrid[nai].velocity.y);
    if (h < -heatOverhead) h = -heatOverhead;
    h = (density * (1 + heatExapnsion * h));
    /*if (abs(h) < minHeat) {
        if (h > 0) h = minHeat;
        if (h < 0) h = -minHeat;
    }*/
    float pn = (dgrid[aj].pressure + dgrid[naj].pressure + dgrid[ai].pressure+ dgrid[nai].pressure) - h*d;// /DT*0.5

    dgrid[I].pressure = (pn) / 4 / ItersCount;
    BoundValues(dgrid);
}
__global__ void DiffuseKernel(Cell* dgrid, int ItersCount = 1) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;

    if (dgrid[I].solidness > 0.f) return;

    int i = (int)((float)I / GridResolutionX);
    int j = I % GridResolutionX;

    float DX = 1.f / GridResolutionX;
    float DY = 1.f / GridResolutionY;

    int aj = I + 1, naj = I - 1;
    int ai = I + GridResolutionY, nai = I - GridResolutionX;
    if (IsOnBound(i, j)) return;

    float h = (dgrid[aj].heat + dgrid[naj].heat) + (dgrid[ai].heat + dgrid[nai].heat) - 4.f * dgrid[I].heat;
    if (h < -heatOverhead) h = -heatOverhead;
    h = (density * (1 + heatExapnsion * h));
    /*if (abs(h) < minHeat) {
        if (h > 0) h = minHeat;
        if (h < 0) h = -minHeat;
    }*/

    dgrid[I].velocity.x += -DT / (h*1.f) * (dgrid[aj].pressure - dgrid[naj].pressure)  / ItersCount * 1.f;
    dgrid[I].velocity.y += -DT / (h*1.f) * (dgrid[ai].pressure - dgrid[nai].pressure)  / ItersCount * 1.f;
    BoundValues(dgrid);
    
}
__global__ void ForcesKernel(Cell* dgrid,int mouseButton,Vector3f mousePos,Vector3f mouseMovement) {
    int I = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)((float)I / GridResolutionX);
    int j = I % GridResolutionX;

    float DX = 1.f / GridResolutionX;
    float DY = 1.f / GridResolutionY;

    int aj = I + 1, naj = I - 1;
    int ai = I + GridResolutionX, nai = I - GridResolutionX;

    float h = (dgrid[aj].heat + dgrid[naj].heat) + (dgrid[ai].heat + dgrid[nai].heat) - 4.f * dgrid[I].heat;
    //if (abs(h) > 4.f) dgrid[I].solidness = 1.f;
    dgrid[I].heat += h * heat_diffusion * DT;

    if (dgrid[I].solidness > 0.f) {
        //dgrid[I].heat += 0.1f*DT;
    }

    if (IsOnBound(i, j)) {
        if (i == 0) { dgrid[I].velocity = -dgrid[ai].velocity; dgrid[I].pressure = dgrid[ai].pressure; dgrid[I].color = dgrid[ai].color; }
        if (i == GridResolutionY-1) { dgrid[I].velocity = -dgrid[nai].velocity; dgrid[I].pressure = dgrid[nai].pressure; dgrid[I].color = dgrid[nai].color; }
        if (j == 0) { dgrid[I].velocity = -dgrid[aj].velocity; dgrid[I].pressure = dgrid[aj].pressure; dgrid[I].color = dgrid[aj].color; }
        if (j == GridResolutionX-1) { dgrid[I].velocity = -dgrid[naj].velocity; dgrid[I].pressure = dgrid[naj].pressure; dgrid[I].color = dgrid[naj].color; }
        /*if (i == 0) { dgrid[I].velocity = dgrid[ai].velocity; dgrid[I].pressure = dgrid[ai].pressure; dgrid[I].color = dgrid[ai].color*0; }
        if (i == GridResolutionY - 1) { dgrid[I].velocity = dgrid[nai].velocity; dgrid[I].pressure = dgrid[nai].pressure; dgrid[I].color = dgrid[nai].color*0; }
        if (j == 0) { dgrid[I].velocity = dgrid[aj].velocity; dgrid[I].pressure = dgrid[aj].pressure; dgrid[I].color = dgrid[aj].color*0; }
        if (j == GridResolutionX - 1) { dgrid[I].velocity = dgrid[naj].velocity; dgrid[I].pressure = dgrid[naj].pressure; dgrid[I].color = dgrid[naj].color*0; }*/

        return;
    }
    if (dgrid[I].solidness > 0.f) {
        float M = ((1.f - dgrid[aj].solidness) + (1.f - dgrid[naj].solidness) + (1.f - dgrid[ai].solidness) + (1.f - dgrid[nai].solidness));
        if (M == 0) return;
        dgrid[I].velocity = -((dgrid[aj].velocity * (1.f - dgrid[aj].solidness) + dgrid[naj].velocity * (1.f - dgrid[naj].solidness)) + (dgrid[ai].velocity * (1.f - dgrid[ai].solidness) + dgrid[nai].velocity * (1.f - dgrid[nai].solidness))) / M;
        dgrid[I].pressure = (dgrid[aj].pressure * (1.f - dgrid[aj].solidness) + dgrid[naj].pressure * (1.f - dgrid[naj].solidness) + dgrid[ai].pressure * (1.f - dgrid[ai].solidness) + dgrid[nai].pressure * (1.f - dgrid[nai].solidness)) / M;
        dgrid[I].color = ((dgrid[aj].color * (1.f - dgrid[aj].solidness) + dgrid[naj].color * (1.f - dgrid[naj].solidness)) + (dgrid[ai].color * (1.f - dgrid[ai].solidness) + dgrid[nai].color * (1.f - dgrid[nai].solidness))) / M;
        return;
    }

    Vector3f newVelocity = Vector3f(dgrid[I].velocity.x, dgrid[I].velocity.y);

    /*Vector3f delta1 = Vector3f((i - GridResolutionY / 2) * 1, j-100);
    Vector3f delta2 = Vector3f(i - GridResolutionY / 2, j - GridResolutionX / 2 - 100);
    float dst1 = delta1.Magnitude();
    float dst2 = delta2.Magnitude();
    float v1 = 1.f / (1.f + powf(.28f * dst1, 2.f));
    float v2 = 1.f / (1.f + powf(.28f * dst2, 2.f));*/

    //newVelocity.x += (v1* velocity_generation_intens) * DT;
    //newVelocity.x += -v2 * velocity_generation_intens * DT;
    //dgrid[I].heat += (v1* pressure_generation_intens -dgrid[I].heat) * 0.05f * DT;

    //printf("%f\n",(float)cosf(1.f));
    //dgrid[I].color.x += v1 * color_generation_intens * DT;
    //dgrid[I].color.z += v2 * color_generation_intens * DT;

    int s = 80;
    Vector3f offset = Vector3f(-00.f, -s / 2);
    if (i > GridResolutionY / 2 + offset.y && i < GridResolutionY / 2 + s + offset.y &&
        j > GridResolutionX / 2 + offset.x-200 && j < GridResolutionX / 2 + s + offset.x-200) {

        //newVelocity.x += velocity_generation_intens * DT;
        //dgrid[I].color.x += color_generation_intens * DT;
        //dgrid[I].pressure += pressure_generation_intens*DT;
    }
    if (i > GridResolutionY / 2 + offset.y && i < GridResolutionY / 2 + s + offset.y &&
        j > GridResolutionX / 2 + offset.x+200 && j < GridResolutionX / 2 + s + offset.x+200) {

        //newVelocity.x += -velocity_generation_intens * DT;
        //dgrid[I].color.z += color_generation_intens * DT;
        //dgrid[I].pressure -= pressure_generation_intens * DT;
    }

    //newVelocity.y += .5f*DT;

    /*Vector3f uv = Vector3f((float)j / GridResolutionX, (float)i / GridResolutionY);
    if (j < 100) {
        newVelocity.x += 8.1f;
    }*/

    float n1 = GridResolutionX * .2f / (1.f + powf(Vector3f(.5 - (float)j / GridResolutionX, .2 - (float)i / GridResolutionY).Magnitude() * .8f * GridResolutionX, 2.f));
    float n2 = GridResolutionX * .2f / (1.f + powf(Vector3f(.5 - (float)j / GridResolutionX, .7 - (float)i / GridResolutionY).Magnitude() * .3f * GridResolutionX, 2.f));

    //newVelocity += Vector3f(0, 1.f) * n1 * .8;
    //newVelocity += Vector3f(0, -1.f) * n2 * .8;

    //dgrid[I].color.z += n1 * .0025f;
    //dgrid[I].color.x += n2 * .0025f;
    
    //dgrid[I].heat += n1 * -0.001f;
    //dgrid[I].heat += n2 * 0.001f;

    float Delta = max(((float)i / GridResolutionY)-.8f,0.f);
    //dgrid[I].heat += n2 * 0.006f * DT;

    //if (dgrid[I].heat > 10.f) dgrid[I].heat = 10.f;
    //if (dgrid[I].heat < -10.f) dgrid[I].heat = -10.f;

    Vector3f mouseDelta = Vector3f(mousePos.x-(float)j/GridResolutionX, mousePos.y - (float)i / GridResolutionY);
    float mouseMag = mouseDelta.Magnitude();
    float normalDistr = GridResolutionX*.2f / (1.f + powf(mouseMag*1.f*GridResolutionX, 2.f));
    const float MF = 20.f;
    if (mouseButton == 0 || mouseButton == 2) {
        newVelocity += mouseMovement * normalDistr * MF;
        if (mouseButton == 0) {
            //dgrid[I].heat += 0.05f* normalDistr *DT;
            dgrid[I].color.x += mouseMovement.Magnitude() * normalDistr * .05f ;
        } else if (mouseButton == 2) {

            //dgrid[I].heat -= 0.001f* normalDistr * DT;
            dgrid[I].color.z += mouseMovement.Magnitude() * normalDistr * .05f ;
        }
    } else if (mouseButton > 0) {
        if (mouseMag < 0.05f && mouseMag > 0.f) {
            dgrid[I].color.y += normalDistr * .1f;
            newVelocity += -mouseDelta/ mouseMag* normalDistr * 30.2f;
        }
    }

    dgrid[I].velocity = newVelocity;
}


template<class T>
__host__ T hlerp(T a, T b, float t) {
    return a + (b - a) * t;
}
int2 GetPUCount() {
    int blocks = max((int)((float)(GridResolutionX * GridResolutionY) / 1024), 1);
    int threads = min(GridResolutionX * GridResolutionY, 1024);
    return make_int2(blocks,threads);
}
void update() {
    int2 PUCount = GetPUCount();
    for (int i = 0; i < 4; i++) {
        //CalculateVortice << < PUCount.x, PUCount.y >> > (d_grid);
        AdvectionKernel <<< PUCount.x, PUCount.y >>> (d_grid);
        gpuErrchk(cudaPeekAtLastError());
        for (int t = 0; t < 1; t++) {
            PressureKernel << < PUCount.x, PUCount.y >> > (d_grid, 1);
            gpuErrchk(cudaPeekAtLastError());
            DiffuseKernel << < PUCount.x, PUCount.y >> > (d_grid, 1);
            gpuErrchk(cudaPeekAtLastError());
        }
        ForcesKernel << < PUCount.x, PUCount.y >> > (d_grid, mouseButton, mousePos, mouseMovement);
        gpuErrchk(cudaPeekAtLastError());
    }
    //printf("pos %f %f\n", mouseMovement.x, mouseMovement.y);
}

int RenderState = 2;
float RenderIntens = 1.f;
void RenderString(float x, float y, const unsigned char* string, Vector3f const& rgb = Vector3f(1.f,1.f,1.f))
{
    char* c;

    glColor3f(rgb.x, rgb.y, rgb.z);
    glRasterPos2f(x, y);

    glutBitmapString(GLUT_BITMAP_HELVETICA_18, string);
}
vector<unsigned char> normalizeGrid() {
    //unsigned char image[GridResolution*GridResolution*4];
    vector<unsigned char> image;
    image.resize(GridResolution * GridResolution * 4);

    for (int i = 0; i < GridResolutionY; i += ResolutionScale) {
        for (int j = 0; j < GridResolutionX; j += ResolutionScale) {
            Vector3f v = hlerp(Vector3f(),fire_color, h_grid[i * GridResolution + j].heat)* RenderIntens;
            /*image[i * GridResolution * 4 + j * 4 + 0] = h_grid[i * GridResolution + j].color.x * 255;
            image[i * GridResolution * 4 + j * 4 + 1] = h_grid[i * GridResolution + j].color.y * 255;
            image[i * GridResolution * 4 + j * 4 + 2] = h_grid[i * GridResolution + j].color.z * 255;*/
            image[i * GridResolution * 4 + j * 4 + 0] = (unsigned char)max(min(v.x, 255.f), 0.f);
            image[i * GridResolution * 4 + j * 4 + 1] = (unsigned char)max(min(v.y, 255.f), 0.f);
            image[i * GridResolution * 4 + j * 4 + 2] = (unsigned char)max(min(v.z, 255.f), 0.f); //(unsigned char)max(255 - h_grid[i * GridResolution + j].heat * 255*.125f, 0.f)
            image[i * GridResolution * 4 + j * 4 + 3] = 255;
        }
    }
    return image;
}
bool isRecording = false;
int recordFrame = 0;
void display() {
    clock_t start = clock();
    if (!paused) {
        update();
        //paused = true;
    }

    const int frameSkipRate = 20;
    if (frame % frameSkipRate == 0) {
        auto renderstart = chrono::high_resolution_clock::now();
        cudaMemcpyAsync(h_grid, d_grid, sizeof(Cell)*GridResolutionX*GridResolutionY,cudaMemcpyDeviceToHost);
        glViewport(0, 0, 750, 750);
        glClear(GL_COLOR_BUFFER_BIT);

        if (isRecording) {
            vector<unsigned char> grid = normalizeGrid();
            lodepng::encode("video/frame_" + to_string(recordFrame) + ".png", grid, (unsigned int)GridResolution, (unsigned int)GridResolution);
            printf("frame: %i (%.1f : %.1f time)\n", recordFrame, (float)recordFrame / 30 / 60, (float)(recordFrame % (30 * 60)) / 30);
            recordFrame++;


            glBegin(GL_POINTS);
            for (int i = 0; i < GridResolutionY; i += ResolutionScale) {
                for (int j = 0; j < GridResolutionX; j += ResolutionScale) {
                    Vector3f cell = Vector3f(grid[i * GridResolutionX + j+0], grid[i * GridResolutionX + j+1], grid[i * GridResolutionX + j+2]);
                    
                    glColor3f(cell.x/255,cell.y/255,cell.z/255);
                    glVertex2f(((float)j / (GridResolutionX - 1) - 0.5f) * 2, -((float)i / (GridResolutionY - 1) - 0.5f) * 2);
                }
            }
            glEnd();
        } else {

            Vector3f clr = Vector3f();


            glBegin(GL_POINTS);

            for (int i = 0; i < GridResolutionY; i += ResolutionScale) {
                for (int j = 0; j < GridResolutionX; j += ResolutionScale) {
                    int flatIndex = i * GridResolutionX + j;

                    if (h_grid[flatIndex].solidness > 0) {
                        glColor3f(0.5f * (1.f + h_grid[flatIndex].heat), 0.5f * (1.f + h_grid[flatIndex].heat*0.6), 0.5f);
                    }
                    else {
                        //cout << h_grid[flatIndex].color.x << endl;
                        float v = h_grid[flatIndex].velocity.Magnitude() * .3f * RenderIntens;
                        if (RenderState == 0)
                            glColor3f(abs(h_grid[flatIndex].velocity.x) * RenderIntens, 0.f, abs(h_grid[flatIndex].velocity.y * RenderIntens));
                        else if (RenderState == 1)
                            glColor3f(h_grid[flatIndex].pressure * 1.f * RenderIntens, 0.f, -h_grid[flatIndex].pressure * 1.f * RenderIntens);
                        else if (RenderState == 2) {
                            glColor3f(h_grid[flatIndex].color.x * (1.f+h_grid[flatIndex].heat), h_grid[flatIndex].color.y * (1.f + h_grid[flatIndex].heat), h_grid[flatIndex].color.z * (1.f + h_grid[flatIndex].heat));
                        } else if (RenderState == 3) {
                            float d = 0.f;
                            if (flatIndex % GridResolutionX <= 0 || flatIndex % GridResolutionX >= GridResolutionX - 1 ||
                                (int)((float)flatIndex / GridResolutionX) <= 0 || (int)((float)flatIndex / GridResolutionX) >= GridResolutionY - 1) d = 1.f;
                            else d = (h_grid[flatIndex - 1].velocity.x - h_grid[flatIndex + 1].velocity.x) + (h_grid[flatIndex - GridResolutionX].velocity.x - h_grid[flatIndex + GridResolutionX].velocity.x);
                            d *= 1.f * RenderIntens;
                            glColor3f(d, 0.f, -d);
                        }
                        else if (RenderState == 4) {
                            Vector3f clr = hlerp(Vector3f(),fire_color, h_grid[flatIndex].heat)*RenderIntens;
                            glColor3f(clr.x,clr.y,clr.z);
                        }
                        clr += h_grid[flatIndex].color;
                    }
                    glVertex2f(((float)j / (GridResolutionX - 1) - 0.5f) * 2, -((float)i / (GridResolutionY - 1) - 0.5f) * 2);
                }
            }
            glEnd();
        }
        RenderString(-1.f, .89f, (unsigned char*)(("Res: " + to_string(GridResolutionX) + ' ' + to_string(GridResolutionY)).c_str()));
        RenderString(-1.f, .85f, (unsigned char*)(("Clr: " + to_string(RenderIntens)).c_str()));
        //RenderString(-1.f, .81f, (unsigned char*)(("Diff: " + to_string(clr.x) + ',' + to_string(clr.y) + ',' + to_string(clr.z)).c_str()));
        
        RenderString(-1.f, .93f, (unsigned char*)((to_string((int)(float(clock() - start) / CLOCKS_PER_SEC * 1000)) + " delta time").c_str()));
        glFlush();
    }
    glutPostRedisplay();

    cudaError_t err = cudaGetLastError();
    if (err != cudaError_t::cudaSuccess)
        printf("|%s|\n", cudaGetErrorString(err));

    //printf(" delta time: %f, frame: %i\n", (float(clock() - start) / CLOCKS_PER_SEC) * 1000, frame);
    frame++;
}
void keyboard(unsigned char key, int x, int y) {
    printf("%i (%i, %i)\n", key, x, y);

    if (key == 32) {
        paused = !paused;
    }
    if (key == 49) RenderState = 0;
    if (key == 50) RenderState = 1;
    if (key == 51) RenderState = 2;
    if (key == 52) RenderState = 3;
    if (key == 53) RenderState = 4;

    if (key == 45) RenderIntens *= 2.f;
    if (key == 43) RenderIntens *= .5f;
}
void mouseClick(int button, int state, int x, int y) {
    printf("%i\n",state);
    if (state == 1) {
        mouseMovement.x = 0.f;
        mouseMovement.y = 0.f;
        mousePos.x = (float)x / 750;
        mousePos.y = (float)y / 750;
        mouseButton = -1;
    } else {
        mousePos.x = (float)x / 750;
        mousePos.y = (float)y / 750;
        mouseButton = button;
    }
}
void mouseMove(int x, int y) {
    Vector3f uv = Vector3f((float)x / 750.f, (float)y / 750.f);

    mouseMovement.x = uv.x - mousePos.x;
    mouseMovement.y = uv.y - mousePos.y;

    //mousePos = uv;
}
void DrawLine(Vector3f p1,Vector3f p2) {
    p1 = p1 * GridResolutionX;
    p2 = p2 * GridResolutionY;

    Vector3f delta = Vector3f(p2.x-p1.x, p2.y - p1.y);
    float dst = sqrtf(delta.x* delta.x + delta.y* delta.y);

    for (float  t = 0; t <= 1; t+=1.f/GridResolutionX) {
        int2 fpos = make_int2((int)(p1.x+delta.x*t), (int)(p1.y + delta.y * t));
        
        h_grid[fpos.y*GridResolutionX + fpos.x].solidness = 1;
    }
}
void DrawRect(Vector3<int> p1, Vector3<int> p2) {
    Vector3i min = Vector3i(p1.x <= p2.x ? p1.x : p2.x, p1.y <= p2.y ? p1.y : p2.y);
    Vector3i max = Vector3i(p1.x > p2.x ? p1.x : p2.x, p1.y > p2.y ? p1.y : p2.y);
    //Vector3<int> delta = p2-p1;
    printf("Min: %i %i\n", min.x, min.y);
    printf("Max: %i %i\n", max.x, max.y);
    for (int i = min.x; i <= max.x; i++) {
        h_grid[min.y * GridResolutionX + i].solidness = 1.f- h_grid[min.y * GridResolutionX + i].solidness;
        h_grid[max.y * GridResolutionX + i].solidness = 1.f- h_grid[max.y * GridResolutionX + i].solidness;

        h_grid[(min.y+1) * GridResolutionX + i].solidness = 1.f- h_grid[(min.y + 1) * GridResolutionX + i].solidness;
        h_grid[(max.y-1) * GridResolutionX + i].solidness = 1.f- h_grid[(max.y - 1) * GridResolutionX + i].solidness;
    }
    for (int i = min.y+2; i < max.y; i++) {
        h_grid[i * GridResolutionX + min.x].solidness = 1.f- h_grid[i * GridResolutionX + min.x].solidness;
        h_grid[i * GridResolutionX + max.x].solidness = 1.f- h_grid[i * GridResolutionX + max.x].solidness;

        h_grid[i * GridResolutionX + min.x+1].solidness = 1.f- h_grid[i * GridResolutionX + min.x + 1].solidness;
        h_grid[i * GridResolutionX + max.x-1].solidness = 1.f- h_grid[i * GridResolutionX + max.x - 1].solidness;
    }
}

float _hash(float i) { return (float)((int)(i * 651959.f + 19698.f) % 1682); }
float frac(float v) {return v-(int)v; }
float random(Vector3f point) {
    return frac(sin(point*Vector3f(_hash(0.f), _hash(1.f), _hash(2.f))));
}
float Voronoi(Vector3f uv) {
    Vector3f P = Vector3f((int)(uv.x), (int)(uv.y), (int)(uv.z));
    uv = uv - (Vector3f)P;

    float v = 1.f;
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                Vector3f offset = Vector3f(x,y,z);
                Vector3f rp = Vector3f((float)random(offset+P), (float)random(offset + P), (float)random(offset + P));
                rp /= rp.Magnitude();
                Vector3f delta = uv - (rp+offset);

                float len = delta.Magnitude();
                if (v > len) v = len;
            }
        }
    }

    return 1.f-v;
}
int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    h_grid = new Cell[GridResolutionX* GridResolutionY];

    unsigned width,height,channels;
    vector<unsigned char> image, file;
    lodepng::load_file(file, "grid.png");
    unsigned error = lodepng::decode(image, width, height, "grid.png");
    cout << error << endl;
    if (!error) {
        printf("%i; %i %i\n", image[0], width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < channels; k++) {
                }
                Vector3i g = Vector3i((int)((float)j/width * GridResolution), (int)((float)i/height*GridResolution));
                Vector3f rgb = Vector3f(image[i * width*4 + j * 4 + 0], image[i * width*4 + j * 4 + 1], image[i * width*4 + j * 4 + 2]);
                //if (rgb.x > 50 && rgb.y <= 50 && rgb.z <= 50)
                //    h_grid[g.y * GridResolution + g.x].solidness=1.f;
                //else
                //    h_grid[g.y * GridResolution + g.x].color = Vector3f(rgb)/255;
            }
        }
    } else {
        const int offset = 10;
        for (int i = 0; i < GridResolutionY; i++) {
            for (int j = 0; j < GridResolutionX; j++) {
                float vx = 0.f, vy = 0.f;
                int ind = i * GridResolutionX + j;
                Vector3f uv = Vector3f((float)j / GridResolutionX, (float)i / GridResolutionY);
                if (j < 100) {
                    //h_grid[ind].color = Vector3f(abs(cosf((float)i / GridResolutionY * 6)), 0., abs(sinf((float)i / GridResolutionY * 6)));
                }
            }
        }
    }
    

    //DrawLine(Vector3f(.5f-.1f,.4f), Vector3f(.5f+.1f, .4f));
    //DrawRect(Vector3i(200, 200), Vector3i(230, 260));

    const float S = .08f;
    //DrawLine(Vector3f(.5f+S, .5f-S), Vector3f(.5f-S, .5f-S));
    //DrawLine(Vector3f(.5f- S, .5f-S), Vector3f(.5f-S, .5f+S));
    //DrawLine(Vector3f(.5f-S, .5f+S), Vector3f(.5f+S, .5f+S));
    
    cudaHostRegister(h_grid,sizeof(Cell)*GridResolutionX* GridResolutionY, cudaHostRegisterMapped);
    //cudaHostRegister(h_grid, sizeof(Cell) * GridResolutionX * GridResolutionY, cudaHostRegisterDefault);
    cudaMalloc(&d_grid,sizeof(Cell)*GridResolutionX* GridResolutionY);
    cudaMemcpy(d_grid,h_grid,sizeof(Cell)*GridResolutionX* GridResolutionY,cudaMemcpyHostToDevice);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(750*(GridResolutionX/GridResolutionY), 750);
    int mainWindow = glutCreateWindow("A Simple OpenGL Windows Application with GLUT");

    glClearColor(0, 0, 0, 1);
    glPointSize(750.f/GridResolutionY*2* ResolutionScale);

    glutMouseFunc(&mouseClick);
    glutMotionFunc(&mouseMove);
    glutKeyboardFunc(&keyboard);

    glutDisplayFunc(display);

    /*glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH, GL_NICEST);

    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH, GL_NICEST);*/

    glutMainLoop();

    return 0;
}