# https://github.com/totex/PyOpenGL_tutorials/blob/master/video_10_using_PNGs.py
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import numpy

vertex_shader_source = """
#version 330
in vec3 position;
in vec3 color;

out vec3 newColor;
void main()
{
   gl_Position = vec4(position, 1.0f);
   newColor = color;
}
"""

fragment_shader_source = """
#version 330
in vec3 newColor;

out vec4 outColor;
void main()
{
    outColor = vec4(newColor, 1.0f);
}
"""

def main():
    if not glfw.init():
        print("faild to init glfw")
        return

    window = glfw.create_window(800, 600, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        print("faild to init glfw window")
        return

    glfw.make_context_current(window)

    #            positions        colors
    triangle = [0.0, 0.5, 0.0, 1.0, 0.0, 0.0,  # red
                0.5, -0.5, 0.0, 0.0, 1.0, 0.0,  # green
                -0.5, -0.5, 0.0, 0.0, 0.0, 1.0]  # blue

    triangle = numpy.array(triangle, dtype=numpy.float32)

    vertex_shader = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader = shaders.compileProgram(vertex_shader, fragment_shader)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 72, triangle, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shader, "color")

    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)

    glUseProgram(shader)
    glClearColor(0, 0, 0, 0)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT)

        glDrawArrays(GL_TRIANGLES, 0, 3)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()