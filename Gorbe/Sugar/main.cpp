//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiv�ve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Mikulas Bence
// Neptun : G8MZZ1
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;
using f = float;
const int array_size = 30;


void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 130
    precision highp float;

			in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	out vec2 texcoord;			// output attribute: texture coordinate

			void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
    precision highp float;

			uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

			void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

template <typename T, int max_size>
class SimpleArray {
	T container[max_size];
	int current_size;

public:
	SimpleArray() : current_size(0) {}
	int MaxSize() const { return max_size; }
	int CurrentSize() const { return current_size; }

	void PushBack(T elem) {
		if (current_size != max_size)
			container[current_size++] = elem;	// tesztelve.
	}

	T& operator[] (int n) {
		if (n < 0 || n > current_size - 1)
			throw "over/underflow!";
		else
			return container[n];
	}

	T operator[] (int n) const {
		if (n < 0 || n > current_size - 1)
			throw "over/underflow!";
		else
			return container[n];
	}
};

// 3D coordinatas
struct V {
	f v[4];

	explicit V(f x = 0, f y = 0, f z = 0, f w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	operator f*() { return &v[0]; }

	// konstanssal szorz�s balr�l
	V operator* (const f scalar) const {
		return V(v[0] * scalar, v[1] * scalar, v[2] * scalar, v[3] * scalar);
	}
	// konstanssal szorz�s jobbr�l
	friend V operator* (const f scalar, const V& v) {
		return v * scalar;
	}
	// skal�ris szorz�s
	f operator * (const V& rhs) const {
		return v[0] * rhs.v[0] + v[1] * rhs.v[1] + v[2] * rhs.v[2];
	}
	// vektori�lis szorz�s
	V operator % (const V& rhs) const {
		return V(v[1] * rhs.v[2] - v[2] * rhs.v[1], v[2] * rhs.v[0] - v[0] * rhs.v[2], v[0] * rhs.v[1] - v[1] * rhs.v[0]);
	}

	friend V operator/ (const V& v, const f scalar) {
		return V(v.v[0] / scalar, v.v[1] / scalar, v.v[2] / scalar, v.v[3] / scalar);
	}

	V& operator += (const V& r) {
		v[0] += r.v[0]; v[1] += r.v[1]; v[2] += r.v[2]; v[3] += r.v[3];	return *this;
	}

	friend V operator+ (const V& l, const V& r) {
		return V(l.v[0] + r.v[0], l.v[1] + r.v[1], l.v[2] + r.v[2], l.v[3] + r.v[3]);
	}

	friend V operator- (const V& l, const V& r) {
		return V(l.v[0] - r.v[0], l.v[1] - r.v[1], l.v[2] - r.v[2], l.v[3] - r.v[3]);
	}

	V operator- () const {
		return V(-v[0], -v[1], -v[2], -v[3]);
	}

	f Length() const {
		return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

	V Normal() const {
		f l = Length();
		return V(v[0] / l, v[1] / l, v[2] / l);
	}
};


struct C {
	V v;
	C() { }
	explicit C(const f r, const f g, const f b) : v(r, g, b){ }
	explicit C(const V& v) : v(v) { }
	operator f*() { return &v.v[0]; }
	C operator* (const f scalar) const {
		return C(v * scalar);
	}
	C operator* (const C& rhs) const {
		return C(v.v[0] * rhs.v.v[0], v.v[1] * rhs.v.v[1], v.v[2] * rhs.v.v[2]);
	}
	C operator/ (const C& rhs) const {
		return C(v.v[0] / rhs.v.v[0], v.v[1] / rhs.v.v[1], v.v[2] / rhs.v.v[2]);
	}
	C operator+ (const C& rhs) const {
		return C(v + rhs.v);
	}
	C operator- (const C& rhs) const {
		return C(v - rhs.v);
	}
	C operator+ (const f scalar) const {
		return C(v.v[0] + scalar, v.v[1] + scalar, v.v[2] + scalar);
	}
	C operator- (const f scalar) const {
		return C(v.v[0] - scalar, v.v[1] - scalar, v.v[2] - scalar);
	}

	C friend operator- (const f scalar, const C& c) {
		return C(scalar - c.v.v[0], scalar - c.v.v[0], scalar - c.v.v[0]);
	}

	C& operator += (const C& rhs) {
		v += rhs.v;
		return *this;
	}
	C Saturate() const { return C(v.v[0] < 1 ? v.v[0] : 1, v.v[1] < 1 ? v.v[1] : 1, v.v[2] < 1 ? v.v[2] : 1); }
};


namespace Color {
	const C Red = C(1, 0, 0);
	const C Green = C(0, 1, 0);
	const C Blue = C(0, 0, 1);

	const C Yellow = C(1, 1, 0);
	const C Pink = C(1, 0, 1);
	const C Cyan = C(0, 1, 1);

	const C Orange = C(1, 0.5, 0);
	const C Purple = C(0.5, 0, 1);

	const C Grey = C(0.5, 0.5, 0.5);
	const C DarkGrey = C(0.1, 0.1, 0.1);
	const C White = C(1, 1, 1);
}

// row-major matrix 4x4
struct M {
	f m[4][4];

	M() {}
	M(f m00, f m01, f m02, f m03,
		f m10, f m11, f m12, f m13,
		f m20, f m21, f m22, f m23,
		f m30, f m31, f m32, f m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m02; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m02; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m02; m[3][3] = m33;
	}

	operator f*() { return &m[0][0]; }

	M operator*(const M& right) const {
		M result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++)
					result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	M Scale(const V& vec) const {
		M M(*this);
		M.m[0][0] *= vec.v[0];
		M.m[1][1] *= vec.v[1];
		M.m[2][2] *= vec.v[2];
		return M;
	}

	M RotateZ(f angle) const {
		M M(*this);
		f a = M.m[0][0];
		f b = M.m[0][1];
		f c = M.m[1][0];
		f d = M.m[1][1];
		M.m[0][0] = a*cosf(angle) + c*sinf(angle);
		M.m[0][1] = b*cosf(angle) + d*sinf(angle);
		M.m[1][0] = c*cosf(angle) - a*sinf(angle);
		M.m[1][1] = d*cosf(angle) - b*sinf(angle);
		return M;
	}

	M Translate(const V& vector) const {
		M M(*this);
		M.m[3][0] = vector.v[0];
		M.m[3][1] = vector.v[1];
		M.m[3][2] = vector.v[2];
		return M;
	}

	static M I() { return M(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1); }
};

V operator*(const V& vec, const M& mat) {
	V result;
	for (int j = 0; j < 4; j++) {
		result.v[j] = 0;
		for (int i = 0; i < 4; i++)
			result.v[j] += vec.v[i] * mat.m[i][j];
	}
	return result;
}


struct Light {
	enum LightType { Ambient, Directional, Point } type;
	V pos, dir;
	C color;

	Light() : type(Light::Ambient), color(Color::White) {}
	Light(Light::LightType type, V pos, V dir, C c)
		: type(type), pos(pos), dir(dir.Normal()), color(c) {}
};


struct Ray {
	V origin, dir;
	Ray() {}
	Ray(const V& origin, const V& dir) : origin(origin), dir(dir) {}
};


class Camera {
	V position;
	V direction;
	V up;
	V right;

	void Initialize() {
		right = (direction % up).Normal();
		up = (right % direction).Normal();
	}

public:
	// A kamera felbont�sa �lland� marad!
	const int width;
	const int height;

	V GetPosition() const { return position; }
	V GetDirection() const { return direction; }

	void SetState(const V& p, const V& d, const V& u = V(0, 1, 0)) {
		position = p;
		direction = d.Normal();
		up = u.Normal();
		Initialize();
	}

	Camera(const int w, const int h) : width(w), height(h) {
		SetState(V(20, 20, 20), V(-1, -1, -1));
	}

	Ray GetRay(const int x, const int y) const {
		f tmpx = (f) (x - width / 2.) / (f) (height / 2.);
		f tmpy = (f) (y - height / 2.) / (f) (height / 2.);
		return Ray(position, (direction + (right * tmpx) + (up * tmpy)).Normal());
	}
};

struct Material;
struct Intersection;
struct Scene;

class Object {
	const Material* material;
public:
	Object() {};
	Object(const Material* m) : material(m) {}
	virtual f Intersect(const Ray&) const = 0;
	virtual V Normal(const V&) const = 0;
	virtual const Material* GetMaterial() const {
		return material;
	}
	virtual ~Object() {}
};


struct Intersection {
	Ray ray;		// A sug�r ami a tal�latot okozta
	Object* obj;	// Pointer az objektumra amire a metsz�spontot kaptuk
	f distance;

	Intersection() : obj(NULL) { }
	Intersection(const Ray& r, Object* o, const f d) : ray(r), obj(o), distance(d) { }
	V Position() const { return ray.origin + (ray.dir.Normal() * distance); }
	V SurfaceNormal() const {
		return obj->Normal(Position());
	}
};


struct Material {
	virtual C GetColor(const Scene&, const Intersection&, const int) const = 0;
	virtual ~Material() {}
};


class Scene {
	int rec_limit;
public:
	SimpleArray<Object*, array_size> objects;
	SimpleArray<Light*, array_size> lights;
	Camera& camera;
	const C background;

	Scene(Camera& cam, const C& bg, const int rl = 10) : camera(cam), background(bg), rec_limit(rl) {}

	inline Intersection ShootRay(const Ray& ray) const {
		Intersection ret(ray, NULL, -1);
		for (int i = 0; i < objects.CurrentSize(); ++i) {
			f distance = objects[i]->Intersect(ray);
			if (distance > 0) {
				if (ret.obj == NULL || distance < ret.distance) {
					ret.distance = distance;
					ret.obj = objects[i];
				}
			}
		}
		return ret;
	}

	inline C GetColor(const Ray& ray, const int rl = 0) const {
		if (rl > rec_limit)
			return C();

		Intersection h = ShootRay(ray);
		if (h.obj != NULL)
			return h.obj->GetMaterial()->GetColor(*this, h, rl + 1);
		else
			return background;
	}

	inline bool IsVisible(const Light* light, const Intersection& hit) const {
		Intersection new_hit = ShootRay(Ray(
			hit.Position() + (hit.SurfaceNormal() * 0.001),
			(light->pos - hit.Position()).Normal())
			);
		return new_hit.obj == NULL || (hit.Position() - light->pos).Length() < new_hit.distance;
	}
};

#include <thread>
#include <mutex>
#include <vector>

class RayTracer {
	Camera& camera;	// csak hogy ne kelljen mindenhova ki�rni
public:
	Scene& scene;

	RayTracer(Scene& s) : camera(s.camera), scene(s) {}

	void Trace(C* img) {
		long time = glutGet(GLUT_ELAPSED_TIME);
		for (int i = 0; i < camera.height; i++) {
			for (int j = 0; j < camera.width; j++) {
				img[i * camera.width + j] = scene.GetColor(camera.GetRay(j, i));
			}
		}
		f elapsed = (glutGet(GLUT_ELAPSED_TIME) - time) / 1000.;
		printf("Tracing takes %f s", elapsed);
	}

	void TraceM(C* img, int n = 8) const {
		long time = glutGet(GLUT_ELAPSED_TIME);
		std::vector<std::thread> threads;
		std::mutex m;
		int part = 0;
		for (int i = 0; i < n; ++i) {
			threads.push_back(std::thread([&] () {
				while (true) {
					m.lock();
					if (part < camera.height) {
						int i = part++;
						m.unlock();
						for (int j = 0; j < camera.width; j++)
							img[i * camera.width + j] = scene.GetColor(camera.GetRay(j, i));
					} else {
						m.unlock();
						break;
					}
				}
			}));
		}
		for (auto& t : threads)
			t.join();
		f elapsed = (glutGet(GLUT_ELAPSED_TIME) - time) / 1000.;
		printf("Tracing takes %f s", elapsed);
	}
};


class Plane : public Object {
protected:
	V normal;
	f D;	// Ax + By + Cz + D = 0
public:
	Plane(Material* m, const V& n, const f d = 0) : Object(m), normal(n), D(d) {}

	virtual f Intersect(const Ray& ray) const {
		f d = ray.dir * normal;
		// mivel csak 1 oldal�, ez�rt itt visszat�r�nk
		if (d > -0.001)
			return -1;
		f t = -((normal * ray.origin) + D) / d;
		if (t < 0 || t > 100) return -1;
		return t;
	}

	virtual V Normal(const V&) const {
		return normal;
	}
};


class Triangle : public Object {
	V a, b, c, normal;

public:
	Triangle() {}

	Triangle(const Material* m, const V& a, const V& b, const V& c) : Object(m), a(a), b(b), c(c) {
		V ab = b - a;
		V ac = c - a;
		normal = (ab.Normal() % ac.Normal()).Normal();
	}

	virtual f Intersect(const Ray& ray) const {
		f ray_travel_dist = ((a - ray.origin) * normal) / (ray.dir * normal);
		if (ray_travel_dist < 0 || isnan(ray_travel_dist))
			return -1;

		// Sz�moljuk ki, hogy a sug�r hol metszi a sug�r s�kj�t.
		V plane_intersection = ray.origin + (ray.dir * ray_travel_dist);
		const V& x = plane_intersection;

		V ab = b - a;
		V ax = x - a;
		V bc = c - b;
		V bx = x - b;
		V ca = a - c;
		V cx = x - c;

		if ((ab % ax) * normal >= 0)
			if ((bc % bx) * normal >= 0)
				if ((ca % cx) * normal >= 0)
					return ray_travel_dist;
		return -1;
	}

	virtual V Normal(const V&) const {
		return normal;
	}

};


class Rect : public Object {
	Triangle part1, part2;
public:
	Rect() {}
	Rect(const Material* m, const V& a, const V& b, const V& c, const V& d) : Object(m), part1(nullptr, a, b, c), part2(nullptr, a, c, d) { }

	virtual f Intersect(const Ray& ray) const {
		f distance = part1.Intersect(ray);
		if (distance < 0)
			distance = part2.Intersect(ray);
		return distance;
	}

	virtual V Normal(const V&) const {
		return part1.Normal(V());
	}
};


class Quadratic : public Object {
protected:
	f A, B, C, D, E, F, G, H, I, J;
public:
	Quadratic(const Material* m) : Object(m),
		A(0), B(0), C(0), D(0), E(0), F(0), G(0), H(0), I(0), J(0) {}

	void Streching(const V& v) {
		A = A / (v.v[0] * v.v[0]);
		B = B / (v.v[1] * v.v[1]);
		C = C / (v.v[2] * v.v[2]);
		D = D / (v.v[1] * v.v[2]);
		E = E / (v.v[0] * v.v[2]);
		F = F / (v.v[0] * v.v[1]);
		G = G / v.v[0];
		H = H / v.v[1];
		I = I / v.v[2];
	}

	void Translation(const V& v) {
		J = A * v.v[0] * v.v[0] + B * v.v[1] * v.v[1] + C * v.v[2] * v.v[2] +
			D * v.v[1] * v.v[2] + E * v.v[0] * v.v[2] + F * v.v[0] * v.v[1] -
			G * v.v[0] - H * v.v[1] - I * v.v[2] + J;
		G = -2 * A * v.v[0] - E * v.v[2] - F * v.v[1] + G;
		H = -2 * B * v.v[1] - D * v.v[2] - F * v.v[0] + H;
		I = -2 * C * v.v[2] - D * v.v[1] - E * v.v[0] + I;
	}

	virtual f Intersect(const Ray& ray) const {
		const f& px = ray.origin.v[0];
		const f& py = ray.origin.v[1];
		const f& pz = ray.origin.v[2];
		const f& dx = ray.dir.v[0];
		const f& dy = ray.dir.v[1];
		const f& dz = ray.dir.v[2];

		f a = A * dx * dx + B * dy * dy + C * dz * dz + D * dy * dz + E * dx * dz + F * dx * dy;
		f b = 2 * (A * px * dx + B * py * dy + C * pz * dz) +
			D * (py * dz + pz * dy) + E * (px * dz + pz * dx) + F * (px * dy + py * dx) +
			G * dx + H * dy + I * dz;
		f c = A * px * px + B * py * py + C * pz * pz +
			D * py * pz + E * px * pz + F * px * py + G * px + H * py + I * pz + J;
		f d = b * b - 4 * a * c;

		if (d < 0)
			return -1;
		d = sqrtf(d);
		f t1 = (-b - d) / (2 * a);
		f t2 = (-b + d) / (2 * a);
		return t1 < t2 ? t1 : t2;
	}

	V Normal(const V& p) const {
		return V(
			2 * A * p.v[0] + E * p.v[2] + F * p.v[1] + G,
			2 * B * p.v[1] + D * p.v[2] + F * p.v[0] + H,
			2 * C * p.v[2] + D * p.v[1] + E * p.v[0] + I
		).Normal();
	}

	virtual ~Quadratic() {}
};


class Ellipsoid : public Quadratic {
public:
	Ellipsoid(const Material* m) : Quadratic(m) {
		A = B = C = 1;
		J = -1;
	}

	Ellipsoid(const Material* m, const V& str) : Quadratic(m) {
		A = B = C = 1;
		J = -1;
		Streching(str);
	}
};


class DiffuseMaterial : public Material {
	C color;
public:
	DiffuseMaterial(const C& c) : color(c) {}

	C GetColor(const Scene& scene, const Intersection& hit, const int) const {
		C ret;
		for (int i = 0; i < scene.lights.CurrentSize(); ++i) {
			switch (scene.lights[i]->type) {
				case Light::Ambient: {
						ret += scene.lights[i]->color * color;
						break;
					}
				case Light::Directional: {
						if (scene.IsVisible(scene.lights[i], hit)) {
							f intensity = hit.obj->Normal(hit.Position()) * scene.lights[i]->dir.Normal();
							if (intensity < 0)
								intensity = 0;
							ret += scene.lights[i]->color *  color * intensity;
						}
						break;
					}
				case Light::Point: {
						if (scene.IsVisible(scene.lights[i], hit)) {
							V pos_to_light = scene.lights[i]->pos - hit.Position();
							f attenuation = pow(1 / pos_to_light.Length(), 2);
							f tmp = hit.SurfaceNormal() * pos_to_light.Normal();
							float intensity = tmp > 0 ? tmp : 0;
							ret += scene.lights[i]->color * color * attenuation * intensity;
						}
						break;
					}
			}
		}
		return ret;
	}
};


class SpecularMaterial : public Material {
	C color;
	f shininess;

public:
	SpecularMaterial(const C& c) : color(c) {
		shininess = 32;
	}

	C GetColor(const Scene& scene, const Intersection& hit, const int) const {
		C ret;
		for (int i = 0; i < scene.lights.CurrentSize(); ++i) {
			switch (scene.lights[i]->type) {
				case Light::Ambient: {
						ret += scene.lights[i]->color * color;
						break;
					}
				case Light::Directional: {
						C specular_color;

						if (scene.IsVisible(scene.lights[i], hit)) {
							f intensity = hit.obj->Normal(hit.Position()) * scene.lights[i]->dir.Normal();
							if (intensity < 0)
								intensity = 0;
							specular_color = scene.lights[i]->color * color * intensity;
						}
						ret += specular_color;
						break;
					}
				case Light::Point: {
						if (scene.IsVisible(scene.lights[i], hit)) {
							// Diff�z sz�m�t�s
							V pos_to_light = scene.lights[i]->pos - hit.Position();
							f attenuation = pow(1 / pos_to_light.Length(), 2);
							f tmp = hit.SurfaceNormal() * pos_to_light.Normal();
							float intensity = tmp > 0 ? tmp : 0;
							C specular_color = scene.lights[i]->color * color * attenuation * intensity;
							ret += specular_color;

							// Spekul�ris sz�m�t�s
							V light_dir = pos_to_light.Normal();
							V eye_dir = -hit.ray.dir;
							V halfway_dir = (light_dir + eye_dir.Normal()).Normal();
							V surface_normal = hit.SurfaceNormal();
							f d = halfway_dir * surface_normal;
							d = d > 0 ? d : 0;

							float specular_power = pow(d, shininess);
							ret += (scene.lights[i]->color * specular_power) * (specular_color * 0.2);
						}
						break;
					}
			}
		}
		return ret;
	}
};


class ReflectiveMaterial : public Material {
	const C F0;
	/*
	ReflectiveMaterial(const C& n, const C& k)
		: F0(((n - 1)*(n - 1) + k*k) /
			((n + 1)*(n + 1) + k*k)) {}
	
	*/

	inline V reflect(const V& I, const V& N) const {
		return I - (N * (2.0 * (N * I)));
	}

	inline C Fresnel(f cosTheta) const {
		return F0 + (1 - F0) * pow(1 - cosTheta, 5);
	}

public:
	C GetColor(const Scene& scene, const Intersection& hit, const int rl) const {	
		Ray reflected_ray;
		reflected_ray.dir = reflect(hit.ray.dir, hit.SurfaceNormal());
		reflected_ray.origin = hit.Position() + (reflected_ray.dir * 1e-3);
		return Fresnel(-hit.ray.dir * hit.SurfaceNormal()) * scene.GetColor(reflected_ray, rl + 1);
	}
};


class SmoothMaterial : public Material {
	C F0;
	float n;
public:
	inline V reflect(const V& inDir, const V& normal) const {
		return inDir - normal * (normal * inDir) * 2.0f;
	}
	inline V refract(const V& inDir, const V& normal) const {
		f ior = n;
		f cosa = -(normal * inDir);
		if (cosa < 0)
			cosa = -cosa;
		V normal_inv = -normal;
		ior = 1 / n;

		f disc = 1 - (1 - cosa * cosa) / ior / ior;
		if (disc < 0)
			return reflect(inDir, normal_inv);

		return inDir / ior + normal_inv * (cosa / ior - sqrt(disc));
	}

	inline C Fresnel(const V& inDir, const V& normal) const {
		f cosa = fabs(normal * inDir);
		return F0 + (C(1, 1, 1) - F0) * pow(1 - cosa, 5);
	}

	C GetColor(const Scene& scene, const Intersection& hit, const int rl) const {
		C F = Fresnel(hit.ray.dir, hit.SurfaceNormal());

		Ray reflected_ray;
		reflected_ray.dir = reflect(hit.ray.dir, hit.SurfaceNormal());
		reflected_ray.origin = hit.Position() + (reflected_ray.dir * 1e-3);
		C ret = F * scene.GetColor(reflected_ray, rl + 1);

		Ray refracted_ray;
		refracted_ray.dir = refract(hit.ray.dir, hit.SurfaceNormal());
		refracted_ray.origin = hit.Position() + (refracted_ray.dir * 1e-3);
		ret += F * scene.GetColor(refracted_ray, rl + 1);

		return ret;
	}
};


/*
struct RefractiveMaterial : public ReflectiveMaterial {
	f n, n_rec;

	inline V refract(V I, V N, double n) {
		double k = 1.0 - n * n * (1.0 - dot(N, I) * dot(N, I));
		if (k < 0.0) {
			return V();
		} else {
			return n * I - (n * dot(N, I) + sqrt(k)) * N;
		}
	}

	RefractiveMaterial(float n, V k, V specular_color, float shininess) : ReflectiveMaterial(n, k, specular_color, shininess), n(n), n_rec(1 / n) {}

	V getColor(Intersection inter, const Light* lgts, size_t lgt_num, int recursion_level) {
		if (dot(inter.ray.direction, inter.normal) > 0) {
			inter.normal = -inter.normal;
		}

		Ray reflected;
		reflected.direction = reflect(inter.ray.direction, inter.normal);
		reflected.origin = inter.pos + 1e-3*reflected.direction;

		Color reflectedColor, refractedColor;

		Ray refracted;
		refracted.direction = refract(inter.ray.direction, inter.normal, inter.ray.in_air ? n : n_rec);
		if (!refracted.direction.isNull()) {
			refracted.origin = inter.pos + 1e-3 * refracted.direction;
			refracted.in_air = !inter.ray.in_air;

			V F_vec = F(dot(-inter.ray.direction, inter.normal));
			reflectedColor = F_vec * scene.shootRay(reflected, recursion_level + 1)
				+ getSpecularHighlight(inter, lgts, lgt_num, reflected.travelled_dist, false);
			refractedColor = (1 - F_vec) * scene.shootRay(refracted, recursion_level + 1);
		} else {
			reflectedColor = scene.shootRay(reflected, recursion_level + 1)
				+ getSpecularHighlight(inter, lgts, lgt_num, reflected.travelled_dist, true);
		}

		return reflectedColor + refractedColor;
	}
};

*/
// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
	unsigned int vao, textureId;	// vertex array object id and texture id
public:
	void Create(C image[windowWidth * windowHeight]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		static float vertexCoords[] = { -1, -1,   1, -1,  -1, 1,
			1, -1,   1,  1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
																							   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

																	  // Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
	}
};

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;

// sz�ks�ges dolgok..
Camera camera(windowWidth, windowHeight);
Scene scene(camera, Color::Cyan);
RayTracer rt(scene);

// a kompoz�ci�
SpecularMaterial blue_material(Color::Cyan);
DiffuseMaterial green_material(Color::Green);

SmoothMaterial smooth_material;

Light light(Light::Ambient, V(1.5, 1.5, 1.5), V(-1, -1, -1), Color::Cyan);
Light light2(Light::Point, V(0, 20, 0), V(0, -1, 0), Color::White * 1000);

V poolA(-25, 0, -5);
V poolB(-25, 0, 5);
V poolC(-25, -3, 5);
V poolD(-25, -3, -5);
V poolE(25, 0, -5);
V poolF(25, 0, 5);
V poolG(25, -3, 5);
V poolH(25, -3, -5);

Rect part1(&blue_material, poolA, poolB, poolC, poolD);
Rect part2(&blue_material, poolB, poolF, poolG, poolC);
Rect part3(&blue_material, poolE, poolH, poolG, poolF);
Rect part4(&blue_material, poolA, poolD, poolH, poolE);
Rect part5(&blue_material, poolG, poolH, poolD, poolC);

V waterA(-25, -0.5, -5);
V waterB(-25, -0.5, 5);
V waterC(25, -0.5, 5);
V waterD(25, -0.5, -5);

Rect water(&smooth_material, waterA, waterB, waterC, waterD);

V edgeA(100, 0, 100);
V edgeB(100, 0, 5);
V edgeC(100, 0, -5);
V edgeD(100, 0, -100);
V edgeE(-100, 0, 100);
V edgeF(-100, 0, 5);
V edgeG(-100, 0, -5);
V edgeH(-100, 0, -100);

Rect edge1(&green_material, edgeF, edgeE, edgeB, edgeA);
Rect edge2(&green_material, poolE, poolF, edgeB, edgeC);
Rect edge3(&green_material, edgeC, edgeD, edgeH, edgeG);
Rect edge4(&green_material, poolB, poolA, edgeG, edgeF);

C background[windowWidth * windowHeight];
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
//	static V background[windowWidth * windowHeight];
	
	rt.scene.objects.PushBack(&part1);
	rt.scene.objects.PushBack(&part2);
	rt.scene.objects.PushBack(&part3);
	rt.scene.objects.PushBack(&part4);
	rt.scene.objects.PushBack(&part5);

	rt.scene.objects.PushBack(&water);

	rt.scene.objects.PushBack(&edge1);
	rt.scene.objects.PushBack(&edge2);
	rt.scene.objects.PushBack(&edge3);
	rt.scene.objects.PushBack(&edge4);

	rt.scene.lights.PushBack(&light);
	rt.scene.lights.PushBack(&light2);

	rt.TraceM(background);

	fullScreenTexturedQuad.Create(background);

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0

															  // Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

struct Quaternion {
	f w, i, j, k;

	Quaternion() : w(0.0f), i(1.0f), j(0.0f), k(0.0f) {}
	Quaternion(f w, f i, f j, f k) : w(w), i(i), j(j), k(k) {}

	Quaternion operator * (const Quaternion& q) {
		Quaternion ret;
		ret.w = w * q.w - i * q.i - j * q.j - k * q.k;
		ret.i = w * q.i + i * q.w + j * q.k - k * q.j;
		ret.j = w * q.j - i * q.k + j * q.w + k * q.i;
		ret.k = w * q.k + i * q.j - j * q.i + k * q.w;
		return ret;
	}

	Quaternion Inverse() { return Quaternion(w, -i, -j, -k); }
};


V RotateAroundAxis(const V& point, const V& axis, const f angle) {
	Quaternion vec(0, point.v[0], point.v[1], point.v[2]);
	Quaternion rot;
	rot.w = cosf(angle / 2.0f);
	rot.i = axis.v[0] * -sinf(angle / 2.0f);
	rot.j = axis.v[1] * -sinf(angle / 2.0f);
	rot.k = axis.v[2] * -sinf(angle / 2.0f);

	vec = (rot * vec) * rot.Inverse();

	return V(vec.i, vec.j, vec.k);
}
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
