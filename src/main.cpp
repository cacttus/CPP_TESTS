#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <assert.h>
#include <memory>

//#include <xmmintrin.h>
#include <x86intrin.h>

//------------------------------------------------------------------------------
union vec4 {
  float c[4];
  __m128 row;
};
typedef vec4 quat;
//------------------------------------------------------------------------------
//
class Globals {
public:
  static int64_t getMicroSeconds() {
    int64_t ret;
    std::chrono::nanoseconds ns = std::chrono::high_resolution_clock::now().time_since_epoch();
    ret = std::chrono::duration_cast<std::chrono::microseconds>(ns).count();
    return ret;
  }
  static int64_t getMilliSeconds() {
    return getMicroSeconds() / 1000;
  }
};
float rand01() {
  static std::random_device rd;
  static std::mt19937 e2(rd());
  static std::uniform_real_distribution<> dist(0, 1);
  return dist(e2);
}

void testAndPrint(std::function<void()> f, std::vector<float>& output) {
  std::cout << "Beginning Test" << std::endl;
  int64_t tA = 0, tB = 0;
  tA = Globals::getMicroSeconds();
  {
    f();
  }
  tB = Globals::getMicroSeconds();
  std::cout << "Time: " << (float)((double)(tB - tA) / 1000.0) << "ms" << std::endl;
  if (output.size() > 0) {
    std::cout << "  out[0]: " << output[0] << std::endl;
    std::cout << "  out[len-1]: " << output[output.size() - 1] << std::endl;
  }
}
//------------------------------------------------------------------------------
//SSE TEST
void DotArrays_ref(std::vector<float>& out,
                   const std::vector<vec4>& a,
                   const std::vector<vec4>& b) {
  assert(a.size() == b.size());
  out.resize(0);
  for (auto vi = 0; vi < a.size(); ++vi) {
    float tmp = 0;
    tmp = a[vi].c[0] * b[vi].c[0] +
          a[vi].c[1] * b[vi].c[1] +
          a[vi].c[2] * b[vi].c[2] +
          a[vi].c[3] * b[vi].c[3];
    out.push_back(tmp);
  }
}
void DotArrays_sse_horizontal(std::vector<float>& out,
                              const std::vector<vec4>& a,
                              const std::vector<vec4>& b) {
  //Note: you must enable compiler intrinsics with a -mavx with GCC
  assert(a.size() == b.size());
  out.resize(0);
  float fout = 0;
  for (auto vi = 0; vi < a.size(); ++vi) {
    __m128 v0 = _mm_mul_ps(a[vi].row, b[vi].row);
    __m128 v1 = _mm_hadd_ps(v0, v0);
    __m128 vr = _mm_hadd_ps(v1, v1);
    _mm_store_ss(&fout, vr);
    out.push_back(fout);
  }
}
//the DotArrays_sse example needs transposed vectors so we skip it
void DotArrays_sse_transpose(std::vector<float>& out,
                             const std::vector<vec4>& a,
                             const std::vector<vec4>& b) {
  //These need to be transposed, this isn't possible unless we transpose everything.
  for (auto vi = 0; vi < a.size(); ++vi) {
    //_MM_TRANSPOSE4_PS(va)
  }
}

//------------------------------------------------------------------------------------
//Alloc Test
typedef unsigned char U8;
//Memory allocation alignment.
inline uintptr_t AlignAddress(uintptr_t addr, size_t align) {
  //So I learned .. size_t and uintptr_t are very different. size_t max is defined by
  std::cout << "size_t max = " << SIZE_MAX << std::endl;

  const size_t mask = align - 1;
  assert((align & mask) == 0);
  uintptr_t x = (addr + mask) & ~mask;
  return x;
}
template <typename T>
inline T* AlignPointer(T* ptr, size_t align) {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t addrAligned = AlignAddress(addr, align);
  return reinterpret_cast<T*>(addrAligned);
}
void* AllocAligned(size_t bytes, size_t align) {
  size_t worstCaseBytes = bytes + align - 1;
  U8* pRawMem = new U8[worstCaseBytes];
  return AlignPointer(pRawMem, align);
}

//-------------------------------------------------------------------
//Quaternion test

#define COORD_SYSTEM_LHS
//#define COORD_SYSTEM_RHS

float v3_dot(const vec4& a, const vec4& b) {
  float r = a.c[0] * b.c[0] + a.c[1] * b.c[1] + a.c[2] * b.c[2];
  return r;
}
vec4 v3_normalize(const vec4& a) {
  float len2 = v3_dot(a, a);
  if (len2 > 0) {
    vec4 r{ a.c[0] / len2, a.c[1] / len2, a.c[2] / len2, 0 };
    return r;
  }
  else {
    vec4 r{ 0, 0, 0, 0 };
    return r;
  }
}
quat axis_angle_to_quat(const vec4& aa) {
  float cos_2 = cos(aa.c[3] / 2.0f);
  float sin_2 = sin(aa.c[3] / 2.0f);

  vec4 a_norm = v3_normalize(aa);

  quat q{ a_norm.c[0] * sin_2, a_norm.c[1] * sin_2, a_norm.c[2] * sin_2, cos_2 };
  return q;
}
vec4 v3_add(const vec4& a, const vec4& b) {
  vec4 r{
    a.c[0] + b.c[0],
    a.c[1] + b.c[1],
    a.c[2] + b.c[2],
    0
  };
  return r;
}
vec4 v3_mul_s(const vec4& a, float s) {
  vec4 r{ a.c[0] * s, a.c[1] * s, a.c[2] * s, 0 };
  return r;
}
vec4 v3_cross(const vec4& a, const vec4& b) {
  //a ^ b
#ifdef COORD_SYSTEM_LHS
  vec4 r{
    a.c[1] * b.c[2] - a.c[2] * b.c[1],
    a.c[2] * b.c[0] - a.c[0] * b.c[2],
    a.c[0] * b.c[1] - a.c[1] * b.c[0],
    0
  };
#elif defined(COORD_SYSTEM_RHS)
  vec4 r{
    b.c[1] * a.c[2] - b.c[2] * a.c[1],
    b.c[2] * a.c[0] - b.c[0] * a.c[2],
    b.c[0] * a.c[1] - b.c[1] * a.c[0],
    0
  };
#endif
  return r;
}
quat q_mul(const quat& p, const quat& q) {
  // [ (pS*qV + qS*pV + pV X qV) (pS*qS - pV . qV) ]
  quat r =
    v3_add(
      v3_add(
        v3_mul_s(q, p.c[3]),
        v3_mul_s(p, q.c[3])),
      v3_cross(p, q));

  r.c[3] = p.c[3] * q.c[3] - v3_dot(p, q);

  return r;
}
quat q_inv_fast(const quat& a) {
  //||q|| = 1
  quat r{ -a.c[0], -a.c[1], -a.c[2], a.c[3] };
  return r;
}
float q_mag2(const quat& a) {
  float r = a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2] + a.c[3] * a.c[3];
  return r;
}
quat q_inv_slow(const quat& a) {
  float mag = q_mag2(a);
  quat nv = q_inv_fast(a);
  quat r{ nv.c[0] / mag, nv.c[1] / mag, nv.c[2] / mag, nv.c[3] / mag };
  return r;
}
vec4 q_rotate(const vec4& v, const quat& q) {
  // q*v*q^-1
  vec4 r = q_mul(q_mul(q, v), q_inv_fast(q));
  return r;
}
//-------------------------------------------------------------------

class PoolAllocator {
public:
};
class StackAllocator {
public:
  int n = 0;
  float a;
  float d = 0;
  float c = 3.14159;
  //This function does something that no other ufnction should do.
  //
};
namespace std {
template <class T>
class weak_ref {
public:
  std::weak_ptr<T> _pt;
  weak_ref(std::shared_ptr<T>& t) {
    _pt = t;
  }
  operator bool() const {
    auto s = _pt.lock();
    return s != nullptr;
  }
  std::shared_ptr<T> operator->() const {
    auto s = _pt.lock();
    return s;
  }
};
}  // namespace std
class GContext {
public:
  void print() { std::cout << "Context used" << std::endl; }
};
class GUser {
public:
  std::weak_ref<GContext> ct;
  GUser(std::shared_ptr<GContext>& c) : ct(c) {}
  void doSomething() {
    if (ct) {
      ct->print();
    }
    else {
      std::cout << "Context went away.." << std::endl;
    }
    //vs weak_ptr ..
    //if (auto s = ct.lock()){
    //  s->print()
    //}
  }
};

int main(int argc, char** argv) {
  //-------------------------------------------------------------------
  {
    std::vector<int> squares_100;
    for (auto i = 1; i < 100; ++i) {
      squares_100.push_back(pow(i, 2));
    }
    std::vector<int> comps;
    std::vector<int> actual;
    for (auto i = 1; i < 100; ++i) {
    }
  }
  //-------------------------------------------------------------------
  {
    //Context
    //This is using a custom class weak_ref to make weak_ptr easier to use at the expense of some cycles.
    std::shared_ptr<GContext> ct = std::make_shared<GContext>();
    std::unique_ptr<GUser> gu = std::make_unique<GUser>(ct);
    gu->doSomething();
    ct = nullptr;
    gu->doSomething();
  }
  //-------------------------------------------------------------------
  {
    //Quaternion Test

    //Cross Product Test
    quat PX{ 1, 0, 0, 0 };
    quat PZ{ 0, 0, 1, 0 };
    quat PY_Test = v3_cross(PZ, PX);
    // Must be 0,1,0,0

    //Quaternion multiplication is NOT associative
    quat v{ 1, 0, 0, 0 };
    quat single_r;
    quat vRet = v;
    single_r = axis_angle_to_quat({ 0, 1, 0, (float)M_PI_2 });  //rotate by pi/2 along the +Y axis to oritent to +z
    vRet = q_rotate(vRet, single_r);
    //should be ( 0,0,1 )
    //Attempt to denormalize floating point values.
    for (int i = 0; i < 100003; ++i) {
      vRet = q_rotate(vRet, single_r);
    }

    //Concatenated rotation
    quat v2{ 0, 1, 0, 0 };

    //1. Individual
    quat test_1 = axis_angle_to_quat({ 0, 0, -1, (float)M_PI_2 });
    test_1 = q_rotate(v2, test_1);
    quat test_2 = axis_angle_to_quat({ 0, 1, 0, (float)M_PI_2 });
    test_2 = q_rotate(test_1, test_2);
    quat test_3 = axis_angle_to_quat({ 1, 0, 0, (float)M_PI });
    test_3 = q_rotate(test_2, test_3);
    //(0,1,0) -> (0,0,-1) -> (0, 0, 1)

    //2. Concatenated
    //Note: Quats must be concatenated in reverse order.
    quat concat_r =
      q_mul(
        axis_angle_to_quat({ 1, 0, 0, (float)M_PI }),  //-Z -> +Z (LHS)
        q_mul(
          axis_angle_to_quat({ 0, 1, 0, (float)M_PI_2 }),  //+X -> -Z (LHS)
          axis_angle_to_quat({ 0, 0, -1, (float)M_PI_2 })  //+Y -> +X (LHS)
          ));

    quat vRet2 = v2;
    vRet2 = q_rotate(vRet2, concat_r);
    //** test_3 must equal vRet2 **
    int n = 0;
    n++;
  }
  //------------------------------------------------------------------
  {
    //Align Test
    void* ret = AllocAligned(10, 16);
    void* ret2 = AllocAligned(10, 128);
    void* ret1 = AllocAligned(20, 16);

    for (auto i = 0; i < 1000; i++) {
      //__m128* test_addr = new __m128[1];
      //uintptr_t addr_of_test_addr = reinterpret_cast<uintptr_t>(test_addr);
      //int x = addr_of_test_addr & 0xF;
      uint32_t* test_addr = new uint32_t[1];
      uintptr_t addr_of_test_addr = reinterpret_cast<uintptr_t>(test_addr);
      int x = addr_of_test_addr & 0xF;

      if (x != 0) {
        int y = 0;
        y++;
      }
    }
  }
  //-------------------------------------------------------------------
  {
    //SSE Test
    std::cout << "Allocating" << std::endl;
    //Initialize Data
    const int COUNT = 10'000'000;
    std::vector<vec4> a(COUNT);
    std::vector<vec4> b(COUNT);
    for (int i = 0; i < COUNT; ++i) {
      a[i].c[0] = rand01();
      a[i].c[1] = rand01();
      a[i].c[2] = rand01();
      b[i].c[0] = rand01();
      b[i].c[1] = rand01();
      b[i].c[2] = rand01();
    }

    std::vector<float> out = {};

    a[0].c[0] = 1;
    a[0].c[1] = 2;
    a[0].c[2] = 3;
    b[0].c[0] = 1;
    b[0].c[1] = 2;
    b[0].c[2] = 3;  // 1*1 + 2*2 + 3*3 = 14

    testAndPrint([&]() {
      DotArrays_ref(out, a, b);
    },
                 out);

    testAndPrint([&]() {
      DotArrays_sse_horizontal(out, a, b);
    },
                 out);
  }
  return 0;
}
