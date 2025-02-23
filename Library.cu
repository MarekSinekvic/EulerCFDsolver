#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "Library.cuh";
#include <iostream>;
#include <vector>;

using namespace std;
//using namespace Library;

template<typename T = float>
struct Vector3 {
public:
    T x = 0.f, y = 0.f, z = 0.f;
    __device__ __host__ Vector3(T x = T(), T y = T(), T z = T()) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    __device__ __host__ float Magnitude() {
        return sqrtf(this->x * this->x + this->y * this->y);
    }
    __device__ __host__ Vector3 operator=(Vector3 value) {
        //return Vector3(value.x, value.y, value.z);
        this->x = value.x;
        this->y = value.y;
        this->z = value.z;
        return *this;
    }
    __device__ __host__ Vector3 operator+(Vector3 value) {
        return Vector3(this->x + value.x, this->y + value.y, this->z + value.z);
    }
    __device__ __host__ Vector3 operator+=(Vector3 value) {
        this->x += value.x;
        this->y += value.y;
        this->z += value.z;
        return *this;
    }
    __device__ __host__ Vector3 operator-(Vector3 value) {
        return Vector3(this->x - value.x, this->y - value.y, this->z - value.z);
    }
    __device__ __host__ Vector3 operator-=(Vector3 value) {
        this->x -= value.x;
        this->y -= value.y;
        this->z -= value.z;
        return *this;
    }
    __device__ __host__ Vector3 operator-() {
        return Vector3(-this->x, -this->y, -this->z);
    }

    template<class T>
    __device__ __host__ Vector3 operator*(T value) {
        return Vector3(this->x * value, this->y * value, this->z * value);
    }
    template<class T>
    __device__ __host__ Vector3 operator*=(T value) {
        this->x *= value;
        this->y *= value;
        this->z *= value;
        return *this;
    }
    template<class T>
    __device__ __host__ static friend Vector3 operator*(T value, Vector3 vec) {
        return Vector3(vec.x * value, vec.y * value, vec.z * value);
    }
    template<class T>
    __device__ __host__ Vector3 operator/(T value) {
        return Vector3(this->x / value, this->y / value, this->z / value);
    }
    template<class T>
    __device__ __host__ Vector3 operator/=(T value) {
        this->x /= value;
        this->y /= value;
        this->z /= value;
        return *this;
    }
    template<class T>
    __device__ __host__ static friend Vector3 operator/(T value, Vector3 vec) {
        return Vector3(vec.x / value, vec.y / value, vec.z / value);
    }

    __device__ __host__ float operator*(Vector3 value) {
        return this->x * value.x + this->y * value.y + this->z * value.z;
    }
};
template<class T>
__device__ T lerp(T a, T b, float t) {
    return a + (b - a) * t;
}

template<class C>
struct Quad {
public:
    Quad* parent = nullptr;
    Quad* childs[4] = { nullptr,nullptr,nullptr,nullptr };
    C value;
    Quad(C cell_value) {
        this->value = cell_value;
    }
    vector<Quad*> iterate_childs(void (*lambda)(Quad,int,int) = [](Quad q, int depth, int index) {}, bool (*condition)(Quad,int) = [](Quad q, int child_index) -> bool {return true;}) {
        vector<Quad*> flat_quads = vector<Quad*>();
        vector<Quad*> quads = vector<Quad*>();
        quads.push_back(this);
        int depth = 0, index = 0;
        while (quads.size() > 0) {
            vector<Quad*> new_quads = vector<Quad*>();

            index = 0;
            for (auto& quad : quads) {
                lambda(*quad,depth,index);
                flat_quads.push_back(quad);
                if (!quad->childs[0]) continue;

                //new_quads.insert(new_quads.end(), (quad->childs), (quad->childs) + 4);
                if (condition(*quad, 0)) new_quads.push_back(quad->childs[0]);
                if (condition(*quad, 1)) new_quads.push_back(quad->childs[1]);
                if (condition(*quad, 2)) new_quads.push_back(quad->childs[2]);
                if (condition(*quad, 3)) new_quads.push_back(quad->childs[3]);

                index++;
                //cout << "V : " << quad->childs[0] << ' ' << quad->childs[1] << ' ' << quad->childs[2] << ' ' << quad->childs[3] << " S " << new_quads.size() << endl;
            }
            depth++;

            quads.clear();
            quads.insert(quads.end(), new_quads.begin(), new_quads.end());
        }

        return flat_quads;
    }
    Quad** divide(C new_cell) {
        Quad* q1 = new Quad(new_cell); Quad* q2 = new Quad(new_cell); Quad* q3 = new Quad(new_cell); Quad* q4 = new Quad(new_cell);
        q1->parent = this; q2->parent = this; q3->parent = this; q4->parent = this;
        
        this->childs[0] = q1; this->childs[1] = q2; this->childs[2] = q3; this->childs[3] = q4;

        return this->childs;
    }
    void unite() {
        vector<Quad*> flat_quads = vector<Quad*>();
        flat_quads = this->iterate_childs();
        for (int i = 0; i < flat_quads.size(); i++) {
            flat_quads[i]->childs[0] = nullptr;
            flat_quads[i]->childs[1] = nullptr;
            flat_quads[i]->childs[2] = nullptr;
            flat_quads[i]->childs[3] = nullptr;
        }
        for (int i = flat_quads.size()-1; i >= 1; i--) {delete (flat_quads[i]);}
    }
    int FindParentIndex() {
        if (!this->parent) return -1;
        if (this->parent->childs[0] == this) return 0;
        if (this->parent->childs[1] == this) return 1;
        if (this->parent->childs[2] == this) return 2;
        if (this->parent->childs[3] == this) return 3;
    }
    vector<Quad*> GetSideQuads(int side = 0) {
        if (!this->childs[0]) return vector<Quad*>();
        
        vector<Quad*> targets;
        if (side == 0) targets = iterate_childs([](Quad c, int d, int i) {}, [](Quad c, int child)->bool {if (child == 0 || child == 1) return true; else return false; });
        if (side == 1) targets = iterate_childs([](Quad c, int d, int i) {}, [](Quad c, int child)->bool {if (child == 0 || child == 2) return true; else return false; });
        if (side == 2) targets = iterate_childs([](Quad c, int d, int i) {}, [](Quad c, int child)->bool {if (child == 1 || child == 3) return true; else return false; });
        if (side == 3) targets = iterate_childs([](Quad c, int d, int i) {}, [](Quad c, int child)->bool {if (child == 2 || child == 3) return true; else return false; });

        vector<Quad*> clear_targets;
        for (auto target : targets) {if (!target->childs[0]) clear_targets.push_back(target);}

        return clear_targets;
    }
    Quad** GetNeighbors() {
        Quad *left = nullptr, *top = nullptr, *right = nullptr, *bottom = nullptr;

        int parent_index = FindParentIndex();
        if (parent_index == -1) return nullptr;
        if (parent_index == 0) { right = this->parent->childs[1]; bottom = this->parent->childs[2]; }
        if (parent_index == 1) { left = this->parent->childs[0]; bottom = this->parent->childs[3]; }
        if (parent_index == 2) { right = this->parent->childs[3]; top = this->parent->childs[0]; }
        if (parent_index == 3) { left = this->parent->childs[2]; top = this->parent->childs[1]; }

        auto find_edge = [=](int s1 = 1, int s2 = 3) -> tuple<Quad*, int> {
            int target_index = -1, depth = 0;
            Quad* target = this;
            while (true) {
                if (!target->parent) break;
                target_index = target->FindParentIndex();
                if (target_index == s1 || target_index == s2) break;
                target = target->parent;
                depth++;
            }
            //if (!target->parent) cout << "no boundary" << endl;
            //else cout << "depth ind: " << depth << " | index: " << target_index << endl;

            if (!target->parent) return { nullptr, target_index };
            return { target, target_index };
        };
        if (parent_index == 0 || parent_index == 1) {
            Quad* target; int index;
            tie(target,index) = find_edge(2, 3);
            if (!target) top = nullptr;
            if (index == 2) top = target->parent->childs[0];
            if (index == 3) top = target->parent->childs[2];
        }
        if (parent_index == 2 || parent_index == 3) {
            Quad* target; int index;
            tie(target,index) = find_edge(0, 1);
            if (!target) bottom = nullptr;
            if (index == 0) bottom = target->parent->childs[2];
            if (index == 1) bottom = target->parent->childs[3];
        }
        if (parent_index == 0 || parent_index == 2) {
            Quad* target; int index;
            tie(target,index) = find_edge(1, 3);
            if (!target) left = nullptr;
            if (index == 1) left = target->parent->childs[0];
            if (index == 3) left = target->parent->childs[2];
        }
        if (parent_index == 1 || parent_index == 3) {
            Quad* target; int index;
            tie(target,index) = find_edge(0, 2);
            if (!target) right = nullptr;
            if (index == 0) right = target->parent->childs[1];
            if (index == 2) right = target->parent->childs[3];
        }

        Quad* neighs[4] = {top,left,right,bottom};
        vector<Quad*> sides = left->GetSideQuads(2);
        return neighs;
    }
};