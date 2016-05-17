// HEAD ++

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

using f = float;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

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

			uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
	uniform vec4  wLiPos;       // pos of light source 
	uniform vec3  wEye;         // pos of eye

			in  vec3 vertexPosition;            // pos in modeling space
	in  vec3 vtxNorm;           // normal in modeling space

			out vec3 wNormal;           // normal in world space
	out vec3 wView;             // view in world space
	out vec3 wLight;            // light dir in world space

			void main() {
		gl_Position = vec4(vertexPosition, 1) * MVP; // to NDC

				vec4 wPos = vec4(vertexPosition, 1) * M;
		wLight  = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
		wView   = wEye * wPos.w - wPos.xyz;
		wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
    precision highp float;

			uniform vec3 kd, ks, ka;// diffuse, specular, ambient ref
	uniform vec3 La, Le;    // ambient and point source rad
	uniform float shine;    // shininess for specular ref

			in  vec3 wNormal;       // interpolated world sp normal
	in  vec3 wView;         // interpolated world sp view
	in  vec3 wLight;        // interpolated world sp illum dir
	out vec4 fragmentColor; // output goes to frame buffer

			void main() {
	   vec3 N = normalize(wNormal);
	   vec3 V = normalize(wView);  
	   vec3 L = normalize(wLight);
	   vec3 H = normalize(L + V);
	   float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
	   vec3 color = ka * La + (kd * cost + ks * pow(cosd,shine)) * Le;
	   fragmentColor = vec4(color, 1);
	}
)";

struct vec3 {
	f x, y, z;

	vec3(f x = 0, f y = 0, f z = 0) : x(x), y(y), z(z) {}

	vec3 operator* (const f scalar) const {
		return vec3(x * scalar, y * scalar, z * scalar);
	}

	friend vec3 operator* (const f scalar, const vec3& v) {
		return v * scalar;
	}

	f operator * (const vec3& rhs) const {
		return x * rhs.x + y * rhs.y + z * rhs.z;
	}

	vec3 operator % (const vec3& rhs) const {
		return vec3(y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z, x * rhs.y - y * rhs.x);
	}

	friend vec3 operator+ (const vec3& l, const vec3& r) {
		return vec3(l.x + r.x, l.y + r.y, l.z + r.z);
	}

	friend vec3 operator- (const vec3& l, const vec3& r) {
		return vec3(l.x - r.x, l.y - r.y, l.z - r.z);
	}

	vec3 operator- () const {
		return vec3(-x, -y, -z);
	}

	f Length() const {
		return sqrtf(x * x + y * y + z * z);
	}

	vec3 Normal() const {
		f l = Length();
		return vec3(x / l, y / l, z / l);
	}

	void SetUniform(unsigned shaderProg, char* name) const {
		int location = glGetUniformLocation(shaderProg, name);
		glUniform3f(location, x, y, z);
	}
};

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	operator float*() { return &m[0][0]; }

	void SetUniform(unsigned shaderProg, char* name) const {
		int location = glGetUniformLocation(shaderProg, name);
		glUniformMatrix4fv(location, 1, GL_TRUE, &m[0][0]);
	}

	static mat4 I() { return mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1); }
	static mat4 Scale(const vec3& vec) { return mat4(vec.x, 0, 0, 0, 0, vec.y, 0, 0, 0, 0, vec.y, 0, 0, 0, 0, 1); }
	static mat4 Translate(const vec3& vec) { return mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, vec.x, vec.y, vec.z, 1); }
	static mat4 Rotate(const f angle, const vec3& axis) {
		return mat4( // Rodrigues matrix
			1 - (axis.x * axis.x - 1) * (cosf(angle) - 1), -axis.z * sinf(angle) - axis.x * axis.y * (cosf(angle) - 1), axis.y * sinf(angle) - axis.x * axis.z * (cosf(angle) - 1), 0,
			axis.z * sinf(angle) - axis.x * axis.y * (cosf(angle) - 1), 1 - (axis.y * axis.y - 1) * (cosf(angle) - 1), -axis.x * sinf(angle) - axis.y * axis.z * (cosf(angle) - 1), 0,
			-axis.y * sinf(angle) - axis.x * axis.z * (cosf(angle) - 1), axis.x * sinf(angle) - axis.y * axis.z * (cosf(angle) - 1), 1 - (axis.z * axis.z - 1) * (cosf(angle) - 1), 0,
			0, 0, 0, 1
			);
	}
};

// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	void SetUniform(unsigned shaderProg, char* name) const {
		int location = glGetUniformLocation(shaderProg, name);
		glUniform4f(location, v[0], v[1], v[2], v[3]);
	}
};


class Camera {
	vec3 wEye, wLookat, wVup;
	f fov, asp, fp, bp;

public:
	Camera() : wLookat(-1, 0, 0), wEye(0, -10, 0), wVup(0, 0, 1) {
		fov = M_PI / 3;
		fp = 2;
		bp = 100;
		asp = 1;
	}

	mat4 V() { // view matrix
		vec3 w = (wEye - wLookat).Normal();
		vec3 u = (wVup % w).Normal();
		vec3 v = w % u;
		return mat4::Translate(-wEye) * mat4(
			u.x, v.x, w.x, 0.0f,
			u.y, v.y, w.y, 0.0f,
			u.z, v.z, w.z, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
			);
	}
	mat4 P() { // projection matrix
		f sy = 1 / tan(fov / 2);
		return mat4(
			sy / asp, 0.0f, 0.0f, 0.0f,
			0.0f, sy, 0.0f, 0.0f,
			0.0f, 0.0f, -(fp + bp) / (bp - fp), -1.0f,
			0.0f, 0.0f, -2 * fp * bp / (bp - fp), 0.0f
			);
	}

	vec3 getEye() { return wEye; }

	void Animate(float t) {
		static vec4 org = vec4(wEye.x, wEye.y, wEye.z, 1);
		vec4 new_pos = org * mat4::Rotate(t / 10, vec3(0, 0, 1));
		wEye = vec3(new_pos.v[0], new_pos.v[1], new_pos.v[2]);
	}
};

struct Light {
	vec4 pos;

	void Animate(float t) {
		static vec4 org = pos;
		pos = org * mat4::Rotate(-t, vec3(0, 0, 1));
	}

	Light(const vec4& pos) : pos(pos) {}
	Light() {}
};

Light light(vec4(-1, -3, 0, 1));

// 3D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

void addUniformFloatToShader(const int& shader, float a, const char* name) {
	int location = glGetUniformLocation(shader, name);
	if (location >= 0)
		glUniform1f(location, a);
	else
		printf("uniform float cannot be set\n");
}

struct Texture {
	unsigned int textureId;
	Texture(char* fname) {
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId); // binding
		int width, height;
		std::vector<f> image = LoadTextureImage(fname, width, height); // meg�rni!
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
			0, GL_RGB, GL_FLOAT, &image[0]); //Texture -> OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Bind(unsigned int shaderProg) {
		int samplerUnit = GL_TEXTURE0; // GL_TEXTURE1, ...
		int location = glGetUniformLocation(shaderProg, "samplerUnit");
		glUniform1i(location, samplerUnit);
		glActiveTexture(samplerUnit);
		glBindTexture(GL_TEXTURE_2D, textureId);
	}

	std::vector<f> LoadTextureImage(char* const fname, int const width, int const height) const {
		std::vector<f> img(3 * width * height);

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				img[(i * height + j) * 3] = (float) i / (float) width;
				img[(i * height + j) * 3 + 1] = (float) j / (float) height;
				img[(i * height + j) * 3 + 2] = 0.5f;
			}
		}

		return img;
	}
};

struct VertexData {
	vec3 position, normal;
	f u, v;
};


struct Material {

};

struct RenderState {
	vec3 wEye;
	mat4 M, V, P, Minv;
	Light light;
	Texture* texture;
	Material* material;
};



struct Geometry {
	unsigned int vao, nVtx;
	vec3 scale, rotAxis, pos;
	float rotAngle;

	Geometry() : scale(vec3(1, 1, 1)), rotAxis(0, 0, 0), pos(0, 0, 0), rotAngle(0) {}

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}

	void Draw() {

		mat4 M = mat4::Scale(scale) *
			mat4::Rotate(rotAngle, rotAxis) *
			mat4::Translate(pos.x);
		mat4 Minv = mat4::Translate(-pos) *
			mat4::Rotate(-rotAngle, rotAxis) *
			mat4::Scale(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		mat4 MVP = M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		MVP.SetUniform(shaderProgram, "MVP");
		M.SetUniform(shaderProgram, "M");
		Minv.SetUniform(shaderProgram, "Minv");
		light.pos.SetUniform(shaderProgram, "wLiPos");
		camera.getEye().SetUniform(shaderProgram, "wLiPos");

		vec3(0.75164, 0.60648, 0.22648).SetUniform(shaderProgram, "kd");
		vec3(0.628281, 0.555802, 0.366065).SetUniform(shaderProgram, "ks");
		vec3(0.24725, 0.1995, 0.0745).SetUniform(shaderProgram, "ka");
		vec3(0.01, 0.01, 0.01).SetUniform(shaderProgram, "La");
		vec3(1, 1, 1).SetUniform(shaderProgram, "Le");

		addUniformFloatToShader(shaderProgram, 51.2f, "shine");


		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}
};

struct ParamSurface : public Geometry {
	virtual VertexData GenVertexData(f u, f v) = 0;

	void Create(int N, int M) {
		Geometry::Create();

		nVtx = N * M * 6;
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		std::vector<VertexData> vtxData(nVtx);
		VertexData * pVtx = &vtxData[0];

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

		int stride = sizeof(VertexData), sVec3 = sizeof(vec3);
		glBufferData(GL_ARRAY_BUFFER, nVtx * stride, &vtxData[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0); // AttribArray 0 = POSITION
		glEnableVertexAttribArray(1); // AttribArray 1 = NORMAL
		glEnableVertexAttribArray(2); // AttribArray 2 = UV
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void *) 0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void *) sVec3);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void *) (2 * sVec3));
	}

};

class Sphere : public ParamSurface {
	f radius;
public:
	Sphere(vec3 center, f r) : radius(r) {
		pos = center;
	}

	void Create() {
		ParamSurface::Create(16, 8);  // tessellation level
	}

	VertexData GenVertexData(f u, f v) {
		VertexData vd;
		vd.normal = vec3(cos(u * 2 * M_PI) * sin(v * M_PI),
			sin(u * 2 * M_PI) * sin(v * M_PI),
			cos(v * M_PI));
		vd.position = vd.normal * radius + pos;
		vd.u = u;
		vd.v = v;
		return vd;
	}
};

class Torus : public ParamSurface {
	f R, r;
public:
	Torus(f R, f r) : R(R), r(r) {}

	void Create() {
		ParamSurface::Create(64, 32);
	}

	VertexData GenVertexData(f u, f v) {
		f i = u * 2.0f * M_PI;
		f j = v * 2.0f * M_PI;

		vec3 t(-sinf(j), cosf(j), 0.0f);
		vec3 s(cosf(j) * -sinf(i), sinf(j) * -sinf(i), cosf(i));

		VertexData vd;
		vd.position = vec3(cosf(j) * (R + cosf(i) * r), sinf(j) * (R + cosf(i) * r), sinf(i) * r);
		vd.normal = -(t % s).Normal();
		vd.u = u;
		vd.v = v;
		return vd;
	}
};

/*
class Object {
protected:
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, pos, rotAxis;
	float rotAngle;
public:

	Object() : scale(vec3(1, 1, 1)), pos(vec3(0, 0, 0)), rotAxis(vec3(0, 0, 0)), rotAngle(0),
		shader(nullptr), material(nullptr), texture(nullptr), geometry(nullptr) {}

	virtual void Draw(RenderState state) {
		state.M = mat4::Scale(scale) *
			mat4::Rotate(rotAngle, rotAxis) *
			mat4::Translate(pos);
		state.Minv = mat4::Translate(-pos) *
			mat4::Rotate(-rotAngle, rotAxis) *
			mat4::Scale(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}
	virtual void Animate(float dt) { }
};


class TheTorus : public Object {
	Torus torus;
public:
	TheTorus(Shader* s) : torus(5, 4) {
		shader = s;
		geometry = &torus;
	}
};

class Golyo : public Object {
	Sphere sphere;
public:
	Golyo() : sphere(vec3(-2, 2, 1), 1) {}
};

class Scene {
	Camera camera;
	std::vector<Object*> objects;
	Light light;
	RenderState state;
public:
	void Render() {
		state.wEye = camera.getEye();
		state.V = camera.V;
		state.P = camera.P;
		state.light = light;
		for (Object * obj : objects)
			obj->Draw(state);
	}

	void Animate(float dt) {
		for (Object * obj : objects)
			obj->Animate(dt);
	}

	void addObject(Object* obj) {
		objects.push_back(obj);
	}
};
*/

struct Shader {
	unsigned int shaderProg;

	void Create(const char * vsSrc, const char * fsSrc, const char * fsOuputName) {

		const char* vsAttrNames[] = { "vertexPosition", "vertexNormal" };

		// vertex
		unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &vsSrc, NULL);
		glCompileShader(vs);
		// fragment
		unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fsSrc, NULL);
		glCompileShader(fs);
		// program
		shaderProg = glCreateProgram();
		glAttachShader(shaderProg, vs);
		glAttachShader(shaderProg, fs);
		// binding
		for (int i = 0; i < 2; i++)
			glBindAttribLocation(shaderProg, i, vsAttrNames[i]);
		glBindFragDataLocation(shaderProg, 0, fsOuputName);
		// link
		glLinkProgram(shaderProg);
		checkLinking(shaderProgram);
			shaderProgram = shaderProg;
	}
	virtual void Bind(RenderState& state) {
		glUseProgram(shaderProg);
	}
};
// The virtual world: collection of two objects


Sphere sphere(vec3(-2, 2, 1), 1);
Torus torus(5, 4);

//Scene scene;

Shader defaultShader;
RenderState rs;

// Initialization, create an OpenGL context
void onInitialization() {

	glEnable(GL_DEPTH_TEST); // z-buffer is on
	glDisable(GL_CULL_FACE); // backface culling is off ?????

	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU


	sphere.Create();
	torus.Create();

	char* vertattribs[] = { "vertexPosition", "vertexNormal" };

	char* fragarrtib = "fragmentColor";
	defaultShader.Create(vertexSource, fragmentSource, fragarrtib);
	defaultShader.Bind(rs);


	/*
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
	glBindAttribLocation(shaderProgram, 1, "vertexNormal");    // vertexColor gets values from Attrib Array 1

															   // Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory
																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);

	*/
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	sphere.Draw();
	torus.Draw();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
	light.Animate(sec);
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
