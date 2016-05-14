//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
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
// Nev : 
// Neptun : 
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

#include <vector>

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

		uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

		in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	in vec3 vertexColor;	 // variable input from Attrib Array selected by glBindAttribLocation
	out vec3 color;				// output attribute

		void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
 precision highp float;

		in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

		void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";


const char * vertexSource = R"(
	uniform mat4 M, Minv, MVP; in vec3 vtxPos; in vec3 vtxNorm;
	out vec4 color;

		void main() {
		gl_Position = vec4(vtxPos, 1) * MVP;
		vec4 wPos = vec4(vtxPos, 1) * M;
		vec4 wNormal = Minv * vec4(vtxNorm, 0);
		color = Illumination(wPos, wNormal);
	}
)";

const char * perVertexShader = R"(
	uniform mat4 MVP, M, Minv; // MVP, Model, Model-inverse
	uniform vec4 kd, ks, ka; // diffuse, specular, ambient ref
	uniform vec4 La, Le; // ambient and point sources
	uniform vec4 wLiPos; // pos of light source in world
	uniform vec3 wEye; // pos of eye in world
	uniform float shine; // shininess for specular ref

		in vec3 vtxPos; // pos in modeling space
	in vec3 vtxNorm; // normal in modeling space
	out vec4 color; // computed vertex color

		void main() {
		gl_Position = vec4(vtxPos, 1) * MVP; // to NDC

			vec4 wPos = vec4(vtxPos, 1) * M;
		vec3 L = normalize(wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w);
		vec3 V = normalize(wEye * wPos.w - wPos.xyz);
		vec4 wNormal = Minv * vec4(vtxNorm, 0);
		vec3 N = normalize(wNormal.xyz);
		vec3 H = normalize(L + V);
		float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
		color = ka * La + (kd * cost + ks * pow(cosd, shine)) * Le;
	}
)";

const char * perPixelVertexShader = R"(
	uniform mat4 MVP, M, Minv; // MVP, Model, Model-inverse
	uniform vec4 wLiPos; // pos of light source 
	uniform vec3 wEye; // pos of eye

		in vec3 vtxPos; // pos in modeling space
	in vec3 vtxNorm; // normal in modeling space

		out vec3 wNormal; // normal in world space
	out vec3 wView; // view in world space
	out vec3 wLight; // light dir in world space

		void main() {
		gl_Position = vec4(vtxPos, 1) * MVP; // to NDC

			vec4 wPos = vec4(vtxPos, 1) * M;
		wLight = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
		wView = wEye * wPos.w - wPos.xyz;
		wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
	}
)";

const char * perPixelpixelShader = R"(
	uniform vec3 kd, ks, ka; // diffuse, specular, ambient ref
	uniform vec3 La, Le; // ambient and point source rad
	uniform float shine; // shininess for specular ref

		in vec3 wNormal; // interpolated world sp normal
	in vec3 wView; // interpolated world sp view
	in vec3 wLight; // interpolated world sp illum dir
	out vec4 fragmentColor; // output goes to frame buffer

		void main() {
		vec3 N = normalize(wNormal);
		vec3 V = normalize(wView);
		vec3 L = normalize(wLight);
		vec3 H = normalize(L + V);
		float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
		vec3 color = ka * La + (kd * cost + ks * pow(cosd, shine)) * Le;
		fragmentColor = vec4(color, 1);
	}
)";

// 3D point in homogeneous coordinates
struct V3 {
	f x, y, z;

	V3(f x = 0, f y = 0, f z = 0) : x(x), y(y), z(z) {}

	V3 operator* (const f scalar) const {
		return V3(x * scalar, y * scalar, z * scalar);
	}

	friend V3 operator* (const f scalar, const V3& v) {
		return v * scalar;
	}

	f operator * (const V3& rhs) const {
		return x * rhs.x + y * rhs.y + z * rhs.z;
	}

	V3 operator % (const V3& rhs) const {
		return V3(y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z, x * rhs.y - y * rhs.x);
	}

	friend V3 operator+ (const V3& l, const V3& r) {
		return V(l.x + r.x, l.y + r.y, l.z + r.z;
	}

	friend V3 operator- (const V3& l, const V3& r) {
		return V3(l.x - r.x, l.y - r.y, l.z - r.z);
	}

	V3 operator- () const {
		return V3(-x, -y, -z);
	}

	f Length() const {
		return sqrtf(x * x + y * y + z * z);
	}

	V3 Normal() const {
		f l = Length();
		return V3(x / l, y / l, z / l);
	}
};


struct V4 {
	f v[4];

	V4(f x = 0, f y = 0, f z = 0, f w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	V4 operator*(const M4& mat) {
		V4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
};

// row-major matrix 4x4
struct M4 {
	f m[4][4];
public:
	M4() {}
	M4(f m00, f m01, f m02, f m03,
		f m10, f m11, f m12, f m13,
		f m20, f m21, f m22, f m23,
		f m30, f m31, f m32, f m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	M4 operator*(const M4& right) {
		M4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator f *() { return &m[0][0]; }

	void SetUniform(unsigned shaderProg, char * name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
	}

	M4 Scale(const V3& vec) const {
		M4 M(*this);
		return M * Scaling(vec);
	}

	M4 Translate(const V3& v) const {
		M4 M(*this);
		return M * Translation(v);
	}

	M4 Rotate(f angle, const V3& axis) const {
		M4 M(*this);
		return M * Rotation(angle, axis);
	}

	static M4 I() { return M4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1); }
	static M4 Scaling(const V3& vec) { return M4(vec.x, 0, 0, 0, 0, vec.y, 0, 0, 0, 0, vec.y, 0, 0, 0, 0, 1); }
	static M4 Translation(const V3& vec) {return M4(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, vec.x, vec.y, vec.z, 1);}
	static M4 Rotation(f angle, const V3& w) {
		return M4( // Rodrigues matrix
			1 - (w.x * w.x - 1) * (cosf(angle) - 1), -w.z * sinf(angle) - w.x * w.y * (cosf(angle) - 1), w.y * sinf(angle) - w.x * w.z * (cosf(angle) - 1), 0,
			w.z * sinf(angle) - w.x * w.y * (cosf(angle) - 1), 1 - (w.y * w.y - 1) * (cosf(angle) - 1), -w.x * sinf(angle) - w.y * w.z * (cosf(angle) - 1), 0,
			-w.y * sinf(angle) - w.x * w.z * (cosf(angle) - 1), w.x * sinf(angle) - w.y * w.z * (cosf(angle) - 1), 1 - (w.z * w.z - 1) * (cosf(angle) - 1), 0,
			0, 0, 0, 1
			);
	}
};


struct VertexData {
	V3 position, normal;
	f u, v;
};

struct Geometry {
	unsigned int vao, nVtx;

	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	void Draw() {
		int samplerUnit = GL_TEXTURE0; // GL_TEXTURE1, ...
		int location = glGetUniformLocation(shaderProg, "samplerUnit");
		glUniform1i(location, samplerUnit);
		glActiveTexture(samplerUnit);
		glBindTexture(GL_TEXTURE_2D, texture.textureId);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}
};

struct ParamSurface : Geometry {
	virtual VertexData GenVertexData(f u, f v) = 0;

	void Create(int N, int M) {
		nVtx = N * M * 6;
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		VertexData * vtxData = new VertexData[nVtx], *pVtx = vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				*pVtx++ = GenVertexData((f) i / N, (f) j / M);
				*pVtx++ = GenVertexData((f) (i + 1) / N, (f) j / M);
				*pVtx++ = GenVertexData((f) i / N, (f) (j + 1) / M);
				*pVtx++ = GenVertexData((f) (i + 1) / N, (f) j / M);
				*pVtx++ = GenVertexData((f) (i + 1) / N, (f) (j + 1) / M);
				*pVtx++ = GenVertexData((f) i / N, (f) (j + 1) / M);
			}
		}

		int stride = sizeof(VertexData), sVec3 = sizeof(V3);
		glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0); // AttribArray 0 = POSITION
		glEnableVertexAttribArray(1); // AttribArray 1 = NORMAL
		glEnableVertexAttribArray(2); // AttribArray 2 = UV
		glVertexAttribPointer(0, 3, GL_f, GL_FALSE, stride, (void *) 0);
		glVertexAttribPointer(1, 3, GL_f, GL_FALSE, stride, (void *) sVec3);
		glVertexAttribPointer(2, 2, GL_f, GL_FALSE, stride, (void *) (2 * sVec3));
	}

};

class Sphere : public ParamSurface {
	V3 center;
	f radius;
public:
	Sphere(V3 c, f r) : center(c), radius(r) {
		Create(16, 8); // tessellation level
	}

	VertexData GenVertexData(f u, f v) {
		VertexData vd;
		vd.normal = V3(cos(u * 2 * M_PI) * sin(v * M_PI),
			sin(u * 2 * M_PI) * sin(v * M_PI),
			cos(v * M_PI));
		vd.position = vd.normal * radius + center;
		vd.u = u;
		vd.v = v;
		return vd;
	}
};

class Flag : public ParamSurface {
	f W, H, D, K, phase;
public:
	Flag(f w, f h, f d, f k, f p) : W(w), H(h), D(d), K(k), phase(p) {
		Create(60, 40); // tessellation level
	}

	VertexData GenVertexData(f u, f v) {
		VertexData vd;
		f angle = u * K * M_PI + phase;
		vd.position = V3(u * W, v * H, sin(angle) * D);
		vd.normal = V3(-K * M_PI * cos(angle) * D, 0, W);
		vd.u = u;
		vd.v = v;
	}
};

class Camera {
	friend class Scene;
	V3 wEye, wLookat, wVup;
	f fov, asp, fp, bp;
public:
	M4 V() { // view matrix
		V3 w = (wEye - wLookat).Normal();
		V3 u = (wVup % w).Normal();
		V3 v = w % u;
		return M4::I().Translate(V3(-wEye.x, -wEye.y, -wEye.z)) *
			M4(u.x, v.x, w.x, 0.0f,
				u.y, v.y, w.y, 0.0f,
				u.z, v.z, w.z, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f);
	}
	M4 P() { // projection matrix
		f sy = 1 / tan(fov / 2);
		return M4(sy / asp, 0.0f, 0.0f, 0.0f,
			0.0f, sy, 0.0f, 0.0f,
			0.0f, 0.0f, -(fp + bp) / (bp - fp), -1.0f,
			0.0f, 0.0f, -2 * fp * bp / (bp - fp), 0.0f);
	}
};

void Draw() {
	M4 M = M4::Scaling(scale) *
		M4::Rotation(rotAng, rotAxis) *
		M4::Translation(pos);
	M4 Minv = M4::Translation(pos) *
		M4::Rotation(-rotAngle, rotAxis) *
		M4::Scaling(V3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	M4 MVP = M * camera.V() * camera.P();

	M.SetUniform(shaderProg, "M");
	Minv.SetUniform(shaderProg, "Minv");
	MVP.SetUniform(shaderProg, "MVP");

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, nVtx);
}

struct Texture {
	unsigned int textureId;
	Texture(char* fname) {
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId); // binding
		int width, height;
		f * image = LoadImage(fname, width, height); // megírni!
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
			0, GL_RGB, GL_f, image); //Texture -> OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

class Material {
	// ???
};

struct Light {
	// ???
};

struct RenderState {
	V3 wEye;
	M4 M, V, P, Minv;
	Light light;
	Texture* texture;
	Material* material;
};

class Scene {
	Camera camera;
	std::vector<Object*> objects;
	Light light;
	RenderState state;

public:
	void Render() {
		state.wEye = camera.wEye;
		state.V = camera.V;
		state.P = camera.P;
		state.light = light;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(f dt) {
		for (Object * obj : objects) obj->Animate(dt);
	}
};

class Object {
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
	V3 scale, pos, rotAxis;
	f rotAngle;

public:
	virtual void Draw(RenderState state) {
		state.M = M4::Scaling(scale) * M4::Rotation(rotAngle, rotAxis) * M4::Translation(pos);
		state.Minv = M4::Translation(-pos) * M4::Rotation(-rotAngle, rotAxis) * M4::Scaling(V3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}
	virtual void Animate(f dt) {}


};

struct Shader {
	unsigned int shaderProg;

	void Create(const char * vsSrc, const char * vsAttrNames[], const char * fsSrc, const char * fsOuputName) {
		unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &vsSrc, NULL);
		glCompileShader(vs);
		unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fsSrc, NULL);
		glCompileShader(fs);
		shaderProgram = glCreateProgram();
		glAttachShader(shaderProg, vs);
		glAttachShader(shaderProg, fs);

		for (int i = 0; i < sizeof(vsAttrNames) / sizeof(char *); i++)
			glBindAttribLocation(shaderProg, i, vsAttrNames[i]);

		glBindFragDataLocation(shaderProg, 0, fsOuputName);
		glLinkProgram(shaderProg);
	}
	virtual
		void Bind(RenderState & state) { glUseProgram(shaderProg); }
};

class ShadowShader : public Shader {
	const char * vsSrc = R"(
		uniform mat4 MVP;
	in vec3 vtxPos;

			void main() {
		gl_Position = vec4(vtxPos, 1) * MVP;
	}
	)";

	const char * fsSrc = R"(
		out vec4 fragmentColor;

			void main() {
		fragmentColor = vec4(0, 0, 0, 1);
	}
	)";

public:
	ShadowShader() {
		static
			const char * vsAttrNames[] = { "vtxPos" };
		Create(vsSrc, vsAttrNames, fsSrc, "fragmentColor");
	}

	void Bind(RenderState & state) {
		glUseProgram(shaderProg);
		M4 MVP = state.M * state.V * state.P;
		MVP.SetUniform(shaderProg, "MVP");
	}
};


Camera camera;

// handle of the shader program
unsigned int shaderProgram;


// The virtual world: collection of two objects


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	/*triangle.Create();
	lineStrip.Create();*/

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
	glBindAttribLocation(shaderProgram, 1, "vertexColor"); // vertexColor gets values from Attrib Array 1

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

	//triangle.Draw();
	//lineStrip.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay(); // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) { // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		f cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		f cY = 1.0f - 2.0f * pY / windowHeight;

		glutPostRedisplay(); // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	f sec = time / 1000.0f;				// convert msec to sec
	glutPostRedisplay();					// redraw the scene
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
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE); // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glEnable(GL_DEPTH_TEST); // z-buffer is on
	glDisable(GL_CULL_FACE); // backface culling is off ?????

#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string) : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay); // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
