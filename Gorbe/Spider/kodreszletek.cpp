const char * vertexSource = R "(
	uniform mat4 M, Minv, MVP; in vec3 vtxPos; in vec3 vtxNorm;
	out vec4 color;

	void main() {
		gl_Position = vec4(vtxPos, 1) * MVP;
		vec4 wPos = vec4(vtxPos, 1) * M;
		vec4 wNormal = Minv * vec4(vtxNorm, 0);
		color = Illumination(wPos, wNormal);
	}
)";

const char * perVertexShader = R "(
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

const char * perPixelVertexShader = R "(
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

const char * perPixelpixelShader = R "(
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

struct mat4 {
    float m[4][4];
    mat4(float m00, …, float m33) {…}
    mat4 operator * (const mat4 & right);

    void SetUniform(unsigned shaderProg, char * name) {
        int loc = glGetUniformLocation(shaderProg, name);
        glUniformMatrix4fv(loc, 1, GL_TRUE, & m[0][0]);
    }
};

mat4 Translate(float tx, float ty, float tz) {
    return mat4(1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        tx, ty, tz, 1);
}
mat4 Rotate(float angle, float wx, float wy, float wz) { }

mat4 Scale(sx, sy, sz) { }


struct Geometry {
    unsigned int vao, nVtx;

    Geometry() {
        glGenVertexArrays(1, & vao);
        glBindVertexArray(vao);
    }
    void Draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, nVtx);
    }
};

void Geometry::Draw() {
    int samplerUnit = GL_TEXTURE0; // GL_TEXTURE1, …
    int location = glGetUniformLocation(shaderProg, "samplerUnit");
    glUniform1i(location, samplerUnit);
    glActiveTexture(samplerUnit);
    glBindTexture(GL_TEXTURE_2D, texture.textureId);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, nVtx);
}

struct VertexData {
    vec3 position, normal;
    float u, v;
};

struct ParamSurface: Geometry {
    virtual VertexData GenVertexData(float u, float v) = 0;
    void Create(int N, int M);
};

void ParamSurface::Create(int N, int M) {
    nVtx = N * M * 6;
    unsigned int vbo;
    glGenBuffers(1, & vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    VertexData * vtxData = new VertexData[nVtx], * pVtx = vtxData;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) { 
        	*pVtx++ = GenVertexData((float) i / N, (float) j / M);
        	*pVtx++ = GenVertexData((float)(i + 1) / N, (float) j / M);
        	*pVtx++ = GenVertexData((float) i / N, (float)(j + 1) / M);
        	*pVtx++ = GenVertexData((float)(i + 1) / N, (float) j / M);
        	*pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
        	*pVtx++ = GenVertexData((float) i / N, (float)(j + 1) / M);
        }
    }

    int stride = sizeof(VertexData), sVec3 = sizeof(vec3);
    glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0); // AttribArray 0 = POSITION
    glEnableVertexAttribArray(1); // AttribArray 1 = NORMAL
    glEnableVertexAttribArray(2); // AttribArray 2 = UV
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void * ) 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void * ) sVec3);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void * )(2 * sVec3));
}

class Sphere: public ParamSurface {
    vec3 center;
    float radius;
    public:
        Sphere(vec3 c, float r): center(c), radius(r) {
            Create(16, 8); // tessellation level
        }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.normal = vec3(cos(u * 2 * M_PI) * sin(v * M_PI),
            sin(u * 2 * M_PI) * sin(v * M_PI),
            cos(v * M_PI));
        vd.position = vd.normal * radius + center;
        vd.u = u;
        vd.v = v;
        return vd;
    }
};

class Flag: public ParamSurface {
    float W, H, D, K, phase;
    public:
        Flag(float w, float h, float d, float k, float p): W(w), H(h), D(d), K(k), phase(p) {
            Create(60, 40); // tessellation level
        }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        float angle = u * K * M_PI + phase;
        vd.position = vec3(u * W, v * H, sin(angle) * D);
        vd.normal = vec3(-K * M_PI * cos(angle) * D, 0, W);
        vd.u = u;
        vd.v = v;
    }
};

class Camera {
    vec3 wEye, wLookat, wVup;
    float fov, asp, fp, bp;
    public:
        mat4 V() { // view matrix
            vec3 w = (wEye - wLookat).normalize();
            vec3 u = cross(wVup, w).normalize();
            vec3 v = cross(w, u);
            return Translate(-wEye.x, -wEye.y, -wEye.z) *
                mat4(u.x, v.x, w.x, 0.0 f,
                    u.y, v.y, w.y, 0.0 f,
                    u.z, v.z, w.z, 0.0 f,
                    0.0 f, 0.0 f, 0.0 f, 1.0 f);
        }
    mat4 P() { // projection matrix
        float sy = 1 / tan(fov / 2);
        return mat4(sy / asp, 0.0 f, 0.0 f, 0.0 f,
            0.0 f, sy, 0.0 f, 0.0 f,
            0.0 f, 0.0 f, -(fp + bp) / (bp - fp), -1.0 f,
            0.0 f, 0.0 f, -2 * fp * bp / (bp - fp), 0.0 f);
    }
};

void Draw() {
    mat4 M = Scale(scale.x, scale.y, scale.z) *
        Rotate(rotAng, rotAxis.x, rotAxis.y, rotAxis.z) *
        Translate(pos.x, pos.y, pos.z);
    mat4 Minv = Translate(-pos.x, -pos.y, -pos.z) *
        Rotate(-rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
        Scale(1 / scale.x, 1 / scale.y, 1 / scale.z);
    mat4 MVP = M * camera.V() * camera.P();

    M.SetUniform(shaderProg, “M”);
    Minv.SetUniform(shaderProg, “Minv”);
    MVP.SetUniform(shaderProg, “MVP”);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, nVtx);
}

struct Texture {
    unsigned int textureId;
    Texture(char * fname) {
        glGenTextures(1, & textureId);
        glBindTexture(GL_TEXTURE_2D, textureId); // binding
        int width, height;
        float * image = LoadImage(fname, width, height); // megírni!
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
            0, GL_RGB, GL_FLOAT, image); //Texture -> OpenGL
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
};


class Scene {
    Camera camera;
    vector<Object*> objects;
    Light light;
    RenderState state;

public:
    void Render() {
        state.wEye = camera.wEye;
        state.V = camera.V;
        state.P = camera.P;
        state.light = light;
        for (Object * obj: objects) obj - > Draw(state);
    }

    void Animate(float dt) {
        for (Object * obj: objects) obj - > Animate(dt);
    }
};

class Object {
    Shader * shader;
    Material * material;
    Texture * texture;
    Geometry * geometry;
    vec3 scale, pos, rotAxis;
    float rotAngle;

public:
    virtual void Draw(RenderState state) {
        state.M = Scale(scale.x, scale.y, scale.z) *  Rotate(rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) * Translate(pos.x, pos.y, pos.z);
        state.Minv = Translate(-pos.x, -pos.y, -pos.z) * Rotate(-rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) * Scale(1 / scale.x, 1 / scale.y, 1 / scale.z);
        state.material = material;
        state.texture = texture;
        shader - > Bind(state);
        geometry - > Draw();
    }
    virtual void Animate(float dt) {}
};

struct Shader {
    unsigned int shaderProg;

    void Create(const char * vsSrc, const char * vsAttrNames[], const char * fsSrc, const char * fsOuputName) {
        unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, & vsSrc, NULL);
        glCompileShader(vs);
        unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, & fsSrc, NULL);
        glCompileShader(fs);
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProg, vs);
        glAttachShader(shaderProg, fs);

        for (int i = 0; i < sizeof(vsAttrNames) / sizeof(char * ); i++)
            glBindAttribLocation(shaderProg, i, vsAttrNames[i]);

        glBindFragDataLocation(shaderProg, 0, fsOuputName);
        glLinkProgram(shaderProg);
    }
    virtual
    void Bind(RenderState & state) { glUseProgram(shaderProg); }
};

class ShadowShader: public Shader {
	const char * vsSrc = R "(
	    uniform mat4 MVP;
	    in vec3 vtxPos;

	    void main() {
	    	gl_Position = vec4(vtxPos, 1) * MVP; 
	    }
	)";

	const char * fsSrc = R "(
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
	    mat4 MVP = state.M * state.V * state.P;
	    MVP.SetUniform(shaderProg, "MVP");
	}
};

void onDisplay() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

 
    glutSwapBuffers(); // exchange the two buffers
}

int main(int argc, char * argv[]) {…
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE |
        GLUT_DEPTH);
    glEnable(GL_DEPTH_TEST); // z-buffer is on
    glDisable(GL_CULL_FACE); // backface culling is off
    
}