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
	#version 140
    precision highp float;

							uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

							in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	in vec3 vertexColor;	    // variable input from Attrib Array selected by glBindAttribLocation
	out vec3 color;				// output attribute

							void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 140
    precision highp float;

							in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

							void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// 3D point in homogeneous coordinates
struct V4 {
	f v[4];

	explicit V4(f x = 0, f y = 0, f z = 0, f w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	operator f*() { return &v[0]; }

	friend V4 operator* (f scalar, const V4& v) {
		return V4(v.v[0] * scalar, v.v[1] * scalar, v.v[2] * scalar, v.v[3] * scalar);
	}

	friend V4 operator/ (const V4& v, f scalar) {
		return V4(v.v[0] / scalar, v.v[1] / scalar, v.v[2] / scalar, v.v[3] / scalar);
	}

	friend V4 operator+ (const V4& l, const V4& r) {
		return V4(l.v[0] + r.v[0], l.v[1] + r.v[1], l.v[2] + r.v[2], l.v[3] + r.v[3]);
	}

	friend V4 operator- (const V4& l, const V4& r) {
		return V4(l.v[0] - r.v[0], l.v[1] - r.v[1], l.v[2] - r.v[2], l.v[3] - r.v[3]);
	}
};

// row-major matrix 4x4
struct M4 {
	f m[4][4];

	M4() {}
	M4(f m00, f m01, f m02, f m03,
		f m10, f m11, f m12, f m13,
		f m20, f m21, f m22, f m23,
		f m30, f m31, f m32, f m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m02; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m02; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m02; m[3][3] = m33;
	}

	operator f*() { return &m[0][0]; }

	M4 operator*(const M4& right) const {
		M4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++)
					result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	M4 Scale(const V4& vec) const {
		M4 M(*this);
		M.m[0][0] *= vec.v[0];
		M.m[1][1] *= vec.v[1];
		M.m[2][2] *= vec.v[2];
		return M;
	}

	M4 RotateZ(f angle) const {
		M4 M(*this);
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

	M4 Translate(const V4& vector) const {
		M4 M(*this);
		M.m[3][0] = vector.v[0];
		M.m[3][1] = vector.v[1];
		M.m[3][2] = vector.v[2];
		return M;
	}

	static M4 I() { return M4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1); }
};

V4 operator*(const V4& vec, const M4& mat) {
	V4 result;
	for (int j = 0; j < 4; j++) {
		result.v[j] = 0;
		for (int i = 0; i < 4; i++)
			result.v[j] += vec.v[i] * mat.m[i][j];
	}
	return result;
}

// 2D camera
struct Camera {
	f wCx, wCy;	// center in world coordinates
	f wWx, wWy;	// width and height in world coordinates
public:
	Camera() { Animate(0); }

	// view matrix: translates the center to the origin
	M4 V() const { return M4::I().Translate(V4(-wCx, -wCy)); }

	// projection matrix: scales it to be a square of edge length 2
	M4 P() const { return M4::I().Scale(V4(2 / wWx, 2 / wWy, 1)); }

	// inverse view matrix
	M4 Vinv() const { return M4::I().Translate(V4(-wCx, -wCy, 1)); }

	// inverse projection matrix
	M4 Pinv() const { return M4::I().Scale(V4(wWx / 2, wWy / 2, 1)); }

	void Animate(f t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	f vertexData[100];		// interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		//GLuint vbo;	// vertex/index buffer object
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0,								// attribute array,
			2, 								//components/attribute,
			GL_FLOAT, 						//component type,
			GL_FALSE, 						//normalize?,
			5 * sizeof(float),				//stride
			reinterpret_cast<void*>(0));	//offset

											// Map attribute array 1 to the color data of the interleaved vbo																							
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(f cX, f cY) {
		if (nVertices >= 20) return;

		V4 wVertex = V4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices] = wVertex.v[0];
		vertexData[5 * nVertices + 1] = wVertex.v[1];
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 0; // blue
		nVertices++;
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);	// fix!!
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw(M4 M) const {
		if (nVertices > 0) {
			M4 MVPTransform = M * camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_TRIANGLE_FAN, 0, nVertices);
		}
	}
};

class CatmullRom {
	V4 r[20];
	f t[20];
	int n;
	const int resolution;

	// these may be const lambdas, capturing time or index... (is it possible? when capture happens?)
	V4 v(const int i) {
		if (i == 0 || n - 2)
			return V4(0, 0);
		return ((r[i + 1] - r[i]) / (t[i + 1] - t[i]) + (r[i] - r[i - 1]) / (t[i] - t[i - 1])) / 2;
	}

	V4 a2(const int i) {
		return ((3 * (r[i + 1] - r[i])) / pow(t[i + 1] - t[i], 2)) - ((v(i + 1) + (2 * v(i))) / (t[i + 1] - t[i]));
	}

	V4 a3(const int i) {
		return ((2 * (r[i] - r[i + 1])) / pow(t[i + 1] - t[i], 3)) + ((v(i + 1) + v(i)) / pow(t[i + 1] - t[i], 2));
	}

	// have to go around by cycle...
	int index(float time) {
		int i;
		for (i = 1; i < n; i++)
			if (t[i] > time)
				return i - 1;
	}

public:
	CatmullRom() : n(0), resolution(20) {}

	void AddPoint(const V4& point) {
		if (n == 20)
			return;
		r[n++] = point;
	}

	void addTime() {
		if (n == 20)
			return;

		static long start = glutGet(GLUT_ELAPSED_TIME);

		t[n] = glutGet(GLUT_ELAPSED_TIME) - start;
		n++;
	}

	V4 GetPlace

		void Draw() {
		if (n < 2)
			return;

		glBegin(GL_LINE_STRIP);
		glColor3f(0.f, 1.f, 0.f);
		float t_max = t[n - 1];
		for (float f = 0; f < t_max; f += 1000 / resolution) {
			int i = index(f);
			((pow(f - t[i], 3) * a3(i)) + (pow(f - t[i], 2) * a2(i)) + ((f - t[i]) * v(i)) + r[i]);	// ???
		}
		glEnd();
	}
};

class Star {
	LineStrip line;
	V4 position;
	f size, defaultSize, angle, shininess;
	long startTime, scale_length, rotation_length, timeShift;
	int numberOfVertices;
	Star* CoG;



public:
	Star() : CoG(nullptr), size(1), defaultSize(1), angle(0), shininess(1), startTime(0),
		timeShift(0), scale_length(3000), rotation_length(6000), numberOfVertices(7) {
	}
	Star& SetPosition(V4 _position) { position = _position; return *this; }
	Star& SetShininess(f _shininess) { shininess = _shininess; return *this; }
	Star& SetSize(f _size) { defaultSize = _size; return *this; }
	Star& SetCenterOfGravity(Star* _star) { CoG = _star; return *this; }
	Star& SetNumberOfVertices(int n) { numberOfVertices = n; return *this; }
	Star& SetAnimationParameters(long _shift = 0, long _scale_length = 3000, long _rotation_length = 6000) {
		timeShift = _shift;
		scale_length = _scale_length;
		rotation_length = _rotation_length;
		return *this;
	}
	V4 GetPosition() { return position; }

	void Create() {
		line.Create();
		line.AddPoint(0, 0);
		for (int i = 0; i < numberOfVertices * 2 + 1; ++i) {
			f a = i / (float)(numberOfVertices * 2) * 3.1415 * 2.0;
			f x = sin(a);
			f y = cos(a);

			if (i % 2) {
				x *= 0.5;
				y *= 0.5;
			}
			line.AddPoint(x, y);
		}
	}

	void Animate(long time) {
		if (startTime == 0)	// first call
			startTime = time;

		long duration = time - startTime + timeShift;
		f scale_pulse = sinf((duration % scale_length) / (float)scale_length * M_PI * 2.0);
		f rotation_pulse = sinf((duration % rotation_length) / (float)rotation_length * M_PI * 2.0);

		size = defaultSize + scale_pulse / 100.0;
		angle = rotation_pulse / 5.0;

		// TODO: update position!
	}

	void Draw() const {
		line.Draw(M4::I().Scale(V4(size, size)).Translate(position).RotateZ(angle));
	}
};

class CatmullStar : public Star {
public:
	void Animate(long int time) {
		// pulzál és forog
		Star::Animate(time);

		// Catmulltól megkérdezi az új pozíciót
	}
};

class Scene {
	CatmullStar brigthest;
	Star star1, star2;

public:
	void Create() {
		// create the objects
		brigthest.SetNumberOfVertices(5).Create();
		star1.SetNumberOfVertices(5).Create();
		star2.SetNumberOfVertices(5).Create();

		// set positions and gravity
		brigthest.SetPosition(V4(-2, -5)).SetSize(0.15).SetAnimationParameters(1500);
		star1.SetPosition(V4(4, 1)).SetSize(0.25).SetCenterOfGravity(&brigthest).SetAnimationParameters(2500);
		star2.SetPosition(V4(-6, 3)).SetSize(0.2).SetCenterOfGravity(&brigthest);
	}

	void Animate(long time) {
		brigthest.Animate(time);
		star1.Animate(time);
		star2.Animate(time);
	}

	void Draw() {
		brigthest.Draw();
		star1.Draw();
		star2.Draw();
	}
};

// The virtual world: collection of two objects
Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	scene.Create();

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
	glBindAttribLocation(shaderProgram, 1, "vertexColor");    // vertexColor gets values from Attrib Array 1

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

	scene.Draw();
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
		f cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		f cY = 1.0f - 2.0f * pY / windowHeight;
		//	lineStrip.AddPoint(cX, cY);
		//	lineStrip2.AddPoint(cY, cX);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	f sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
	scene.Animate(time);
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

