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
							uniform vec3 col = vec3(1.0, 1.0, 1.0);
							in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	in vec3 vertexColor;	    // variable input from Attrib Array selected by glBindAttribLocation
	out vec3 color;				// output attribute

								void main() {
		
		//color = vertexColor;														// copy color from input to output
		//color = vec3(col.x, col.y, col.z);
		color = col;
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

	V4 operator* (f scalar) {
		return V4(v[0] * scalar, v[1] * scalar, v[2] * scalar, v[3] * scalar);
	}

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

	float operator*(const V4& r) {
		return v[0] * r.v[0] + v[1] * r.v[1] + v[2] * r.v[2] + v[3] * r.v[3];
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

class CatmullStar;
//V4 CatmullStar::GetPosition();
// 2D camera
struct Camera {
	f wCx, wCy;	// center in world coordinates
	f wWx, wWy;	// width and height in world coordinates
	CatmullStar* cs;
public:
	Camera() { Animate(0); cs = nullptr; }

	// view matrix: translates the center to the origin
	M4 V() const { return M4::I().Translate(V4(-wCx, -wCy)); }

	// projection matrix: scales it to be a square of edge length 2
	M4 P() const { return M4::I().Scale(V4(2 / wWx, 2 / wWy, 1)); }

	// inverse view matrix
	M4 Vinv() const { return M4::I().Translate(V4(wCx, wCy, 1)); }

	// inverse projection matrix
	M4 Pinv() const { return M4::I().Scale(V4(wWx / 2, wWy / 2, 1)); }

	void Animate(f t) {
		wCx = 0;//10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
		if (cs != nullptr) {
			//V4 starPos = cs->GetPosition();
			//wCx = starPos[0];
			//wCy = starPos[1];
		}
	}

	void SetCenter(f _wCx, f _wCy) {
		wCx = _wCx;
		wCy = _wCy;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

template <int max_size, int draw_mode>
class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	f vertexData[5 * max_size];		// interleaved data of coordinates and colors
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
			5 * sizeof(f),					//stride
			reinterpret_cast<void*>(0));	//offset

											// Map attribute array 1 to the color data of the interleaved vbo																							
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(f cX, f cY) {
		if (nVertices >= max_size) return;

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
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(f), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw(M4 M = M4::I(), V4 color = V4(1, 1, 1)) const {
		if (nVertices > 0) {
			M4 MVPTransform = M * camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform);
			else printf("uniform MVP cannot be set\n");

			location = glGetUniformLocation(shaderProgram, "col");
			float col[] = { color[0], color[1], color[2] };
			if (location >= 0) glUniform3fv(location, 1, col);
			else printf("uniform color cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(draw_mode, 0, nVertices);
		}
	}
};

template <int max_size, typename type>
class Ring {
protected:
	type cont[max_size];
	int size;
public:
	Ring() : size(0) {}
	type& operator[](const int i) {
		return 0 <= i ? cont[i % size] : cont[(i + 1) % size + size - 1];
	}
	type monInc(const int i) {
		int circlesAround = i / size;
		if (i < 0) circlesAround--;
		int idxInCircle = (0 <= i ? (i % size) : ((i + 1) % size + size - 1));

		type Diff = cont[size - 1] - cont[0];

		type avg = Diff / (float) size;

		type circle = Diff + avg;

		return cont[idxInCircle] + circlesAround * circle;
	}
	void Push(const type& elem) {
		if (size < max_size)
			cont[size++] = elem;
	}
	int GetActualSize() { return size; }
};

class CatmullRom {
	Ring<20, V4> r;
	Ring<20, f> t;
	int n;

	V4 v(const int i) {
		//return ((r[i + 1] - r[i]) / (t[i + 1] - t[i]) + (r[i] - r[i - 1]) / (t[i] - t[i - 1])) / 2.0f;
		return ((r[i + 1] - r[i]) / (t.monInc(i + 1) - t.monInc(i)) + (r[i] - r[i - 1]) / (t.monInc(i) - t.monInc(i - 1))) * 0.9f;
	}

	V4 a2(const int i) {
		//return ((3.0f * (r[i + 1] - r[i])) / pow(t[i + 1] - t[i], 2)) - ((v(i + 1) + (2 * v(i))) / (t[i + 1] - t[i]));
		return ((3.0f * (r[i + 1] - r[i])) / pow(t.monInc(i + 1) - t.monInc(i), 2)) - ((v(i + 1) + (2 * v(i))) / (t.monInc(i + 1) - t.monInc(i)));
	}

	V4 a3(const int i) {
		//return ((2.0f * (r[i] - r[i + 1])) / pow(t[i + 1] - t[i], 3)) + ((v(i + 1) + v(i)) / pow(t[i + 1] - t[i], 2));
		return ((2.0f * (r[i] - r[i + 1])) / pow(t.monInc(i + 1) - t.monInc(i), 3)) + ((v(i + 1) + v(i)) / pow(t.monInc(i + 1) - t.monInc(i), 2));
	}

	int index(float time) {
		for (int i = 0; i < n + 2; i++)
			if (t.monInc(i) >= time)
				return i - 1;
	}

	V4 GetPoint(const f time) {
		if (n >= 3) {
			int i = index(time);
			//return (pow(time - t[i], 3) * a3(i)) + (pow(time - t[i], 2) * a2(i)) + ((time - t[i]) * v(i)) + r[i];
			return (pow(time - t.monInc(i), 3) * a3(i)) + (pow(time - t.monInc(i), 2) * a2(i)) + ((time - t.monInc(i)) * v(i)) + r[i];
		}
		return V4();
	}

public:
	CatmullRom() : n(0) {	// sample data....
							//r.Push(V4(0.3, 0));
							//r.Push(V4(5, 5));
							//r.Push(V4(5, -5));
							//r.Push(V4(-5, -5));
							//r.Push(V4(3, 0));
							//r.Push(V4(0.3, 0));

							//t.Push(0);
							//t.Push(2.1);
							//t.Push(3.1);
							//t.Push(4.1);
							//t.Push(6.9);
							//t.Push(7.9);
		n = r.GetActualSize();
	}

	void AddPoint(const V4& point, const long& time) {
		r.Push(point);
		t.Push((float) time / 1000.0);
		n = r.GetActualSize();
	}

	V4 GetPosition(long abs_time) {
		if (n >= 3) {
			static long startTime = abs_time;
			// a bejárás összes ideje [s]
			f duration = (t[t.GetActualSize() - 1] - t[0]) * (t.GetActualSize() + 1) / t.GetActualSize();
			// a bejárás pillanatnyi ideje [ms]
			long state_ms = (abs_time - startTime) % (int) (duration * 1000);

			return GetPoint(t[0] + (state_ms / 1000.0f));
		}
		return V4();
	}

	void Draw() {
		if (n >= 3) {
			int resolution = 100;
			LineStrip<1000, GL_LINE_STRIP> line;
			line.Create();

			float t_max = t[t.GetActualSize() - 1];	// az utolsó elem max!
			float around = (t_max - t[0]);

			for (float f = t[0]; f <= t_max + (around / (float) n)*1.1f; f += around / resolution) {
				V4 point = GetPoint(f);
				line.AddPoint(point[0], point[1]);
			}

			line.Draw(M4::I()*camera.V()*camera.P());
		}
	}
};

class Star {
	LineStrip<20, GL_TRIANGLE_FAN> line;
	V4 position;
	V4 velocity;
	V4 acceleration;
	f size, defaultSize, angle, shininess;
	long startTime, scale_length, rotation_length, timeShift;
	int numberOfVertices;
	Star* CoG;

public:
	Star() : CoG(nullptr), size(1), defaultSize(1), angle(0), shininess(1), startTime(0),
		timeShift(0), scale_length(3000), rotation_length(6000), numberOfVertices(7) {
		velocity = V4(0, 0, 0, 0);
		acceleration = V4(0, 0, 0, 0);
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
			f a = i / (float) (numberOfVertices * 2) * 3.1415 * 2.0;
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
		static long StartTime = time;

		long duration = time - startTime + timeShift;
		f scale_pulse = sinf((duration % scale_length) / (float) scale_length * M_PI * 2.0);
		f rotation_pulse = sinf((duration % rotation_length) / (float) rotation_length * M_PI * 2.0);

		size = defaultSize + scale_pulse / 100.0;
		angle = rotation_pulse / 5.0;

		if (CoG != nullptr) {
			//time 0.1 sec
			float Mcat = 10;
			float M = 10;
			float g = 0.00001;

			V4 dir = CoG->GetPosition() - position;
			float rr = dir * dir;
			//F   = m*a
			//F/m = a
			acceleration = dir*Mcat*g / rr;

			velocity = velocity + acceleration*0.1;

			position = position + velocity*0.1;
		}

	}

	void Draw() const {
		line.Draw(M4::I().Scale(V4(size, size)).RotateZ(angle).Translate(position));
	}
};

class CatmullStar : public Star {
	CatmullRom cr;
public:
	void Animate(long time) {
		Star::Animate(time);					// pulzál és forog
		SetPosition(1 * cr.GetPosition(time));	// Catmulltól megkérdezi az új pozíciót
	}

	void Draw() {
		Star::Draw();	// kirajzolja magát
		cr.Draw();		// kirajzolja a görbét is
	}

	CatmullRom* GetCatmullLine() {
		return &cr;
	}
};

struct Scene {
	CatmullStar brightest;
	Star star1, star2;
	CatmullRom* cr;

	long lastSimulatedTime;

	void Create() {
		// create the objects
		brightest.SetNumberOfVertices(5).Create();
		star1.SetNumberOfVertices(5).Create();
		star2.SetNumberOfVertices(5).Create();

		// set positions and gravity
		brightest.SetPosition(V4(-2, -5)).SetSize(0.15).SetAnimationParameters(1500);
		star1.SetPosition(V4(0, 5)).SetSize(0.1).SetCenterOfGravity(&brightest).SetAnimationParameters(2500);
		star2.SetPosition(V4(-4, 3)).SetSize(0.1).SetCenterOfGravity(&brightest);

		cr = brightest.GetCatmullLine();
	}

	void Animate(long time) {
		long elapsedTime = time - lastSimulatedTime;

		int simulationCycles = elapsedTime % 100;
		for (int i = 0; i < simulationCycles; i++) {
			brightest.Animate(time);
			star1.Animate(time);
			star2.Animate(time);
		}
		lastSimulatedTime = time;
	}

	void AddPoint(const V4& point) {
		long time = glutGet(GLUT_ELAPSED_TIME);
		cr->AddPoint(point, time);
	}

	void Draw() {
		brightest.Draw();
		star1.Draw();
		star2.Draw();
	}
};

Scene scene;
static bool cameraLocked = true;

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
// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	if (!cameraLocked) {
		V4 pos = scene.brightest.GetPosition();
		camera.SetCenter(pos[0], pos[1]);
	}

	scene.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 32) cameraLocked = !cameraLocked;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		f cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		f cY = 1.0f - 2.0f * pY / windowHeight;
		V4 click = V4(cX, cY, 0, 1);
		click = click*(camera.Vinv()*camera.Pinv());
		//lineStrip.AddPoint(cX, cY);
		//lineStrip2.AddPoint(cY, cX);
		cX = click[0];
		cY = click[1];
		scene.AddPoint(V4(cX, cY));
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

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

