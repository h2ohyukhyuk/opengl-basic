import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
from PIL import Image

def window_resize(window, width, height):
    glViewport(0, 0, width, height)

def main():

    # initialize glfw
    if not glfw.init():
        return

    w_width, w_height = 800, 600

    #glfw.window_hint(glfw.RESIZABLE, GL_FALSE)

    window = glfw.create_window(w_width, w_height, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)

    #        positions         colors          texture coords
    cube = [-0.5, -0.5,  0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
             0.5, -0.5,  0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

            -0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
             0.5, -0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
             0.5,  0.5, -0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5,  0.5, -0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

             0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
             0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
             0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

            -0.5,  0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
            -0.5, -0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
            -0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

            -0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
             0.5, -0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
             0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
            -0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0,

             0.5,  0.5, -0.5,  1.0, 0.0, 0.0,  0.0, 0.0,
            -0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,
            -0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,
             0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0]

    cube = np.array(cube, dtype = np.float32)

    indices = [ 0,  1,  2,  2,  3,  0,
                4,  5,  6,  6,  7,  4,
                8,  9, 10, 10, 11,  8,
               12, 13, 14, 14, 15, 12,
               16, 17, 18, 18, 19, 16,
               20, 21, 22, 22, 23, 20]

    indices = np.array(indices, dtype= np.uint32)

    vertex_shader = """
    #version 330
    in layout(location = 0) vec3 position;
    in layout(location = 1) vec3 color;
    in layout(location = 2) vec2 textureCoords;

    uniform mat4 view;
    uniform mat4 model;
    uniform mat4 projection;

    out vec3 newColor;
    out vec2 newTexture;
    void main()
    {
        gl_Position = projection * view * model * vec4(position, 1.0f);
        newColor = color;
        newTexture = textureCoords;
    }
    """

    fragment_shader = """
    #version 330
    in vec3 newColor;
    in vec2 newTexture;

    out vec4 outColor;
    uniform sampler2D samplerTexture;
    void main()
    {
        outColor = texture(samplerTexture, newTexture); //* vec4(newColor, 1.0f);
    }
    """
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, cube.itemsize * len(cube), cube, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

    #position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    #color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    #texture
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)


    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load image
    image = Image.open("res/spinks.jpg")
    img_data = np.array(list(image.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glEnable(GL_TEXTURE_2D)

    glUseProgram(shader)

    glClearColor(0.2, 0.3, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    '''
    numpy 는 row major 로 행렬을 사용한다.
    glsl matrix는 column major 로 행렬을 사용한다.
    numpy에서 glsl matrix로 전달되면서 transpose 된다. 
    '''

    model_s = pyrr.matrix44.create_from_scale(pyrr.Vector3([1.0, 2.0, 3.0]))
    model_rx = pyrr.Matrix44.from_x_rotation(np.deg2rad(120))
    model_ry = pyrr.Matrix44.from_y_rotation(np.deg2rad(30))
    model_rz = pyrr.Matrix44.from_z_rotation(np.deg2rad(40))
    model_t = pyrr.matrix44.create_from_translation(pyrr.Vector3([-0.5, -1.0, -4]))  # transpose 되어서 계산된다.

    model = (model_t.T @ model_rz @ model_ry @ model_rx @ model_s).T

    '''
    카메라 위치가 변한 다는 것은 월드 좌표가 역으로 변하는 것과 같은 것이다. 
    '''
    view_rx = pyrr.Matrix44.from_x_rotation(np.deg2rad(0)).T
    view_ry = pyrr.Matrix44.from_y_rotation(np.deg2rad(0)).T
    view_rz = pyrr.Matrix44.from_z_rotation(np.deg2rad(30)).T
    view_t = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 1.0]) * -1.0)
    view = (view_t.T @ view_rz @ view_ry @ view_rx).T

    '''
    카메라에 담기는 공간을 프로젝션하기 좋게 정규화해주는 변환이다.
    버텍스들을 원근변환의 역으로 변환해줌, 즉 멀리있는 물체가 작아지고
    가까이 있는 물체는 커지게함.
    이 상태를 그대로 오쏘고날 프로젝션 하면 됨.
    '''
    def Perspective(fov, aspectRatio, near, far):
        import math
        fn, f_n = far + near, far - near
        r, t = aspectRatio, 1.0 / math.tan(math.radians(fov) / 2.0)

        return np.matrix([[t / r, 0, 0, 0],
                          [0, t, 0, 0],
                          [0, 0, -fn / f_n, -2.0 * far * near / f_n],
                          [0, 0, -1, 0]])

    # transpose 되어서 계산됨.
    projection = pyrr.matrix44.create_perspective_projection_matrix(60.0, w_width / w_height, 0.1, 100.0)
    projection_ = Perspective(60.0, w_width/w_height, 0.1, 100.0).T

    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()