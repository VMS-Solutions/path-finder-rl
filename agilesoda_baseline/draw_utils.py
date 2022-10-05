import cv2
import imageio
import os
import math
import pandas as pd
import numpy as np
from pyvirtualdisplay import Display

import pyglet
from pyglet.gl import *
white_rgb = (1, 1, 1)
black_rgb = (0, 0, 0)
obstacle_rgb = (204/255, 204/255, 204/255)
agent_rgb = (000/255, 51/255, 000/255) # (147/255, 122/255, 219/255)
box_rgb = (153/255, 204/255, 153/255)
height = 10 # 10 * 10
width = 9 
RAD2DEG = 57.29577951308232


class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):
        self.rotation = float(new)
    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

        
class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        
    def render(self, return_rgb_array=False):
        glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen
    
    def add_geom(self, geom):
        self.geoms.append(geom)


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)


class Geom(object):
    def __init__(self):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)


class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()


class Render():
    def __init__(self, screen_width, screen_height, unit, start_point, data_path, log_path):
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.log_path = log_path
        self.unit = unit
        self.movement = []
        self.boxes = []
        self.obstacles = []
        
        self.viewer = Viewer(screen_width, screen_height)
        self.draw_initial(data_path, type_="box")
        self.draw_initial(data_path, type_="obstacles")
        self.update_movement(start_point, 0)

    def draw_initial(self, data_path, type_):
        if type_ == "box":
            position_data = pd.read_csv(os.path.join(data_path, "box.csv"))
            color = box_rgb
        elif type_ == "obstacles":
            position_data = pd.read_csv(os.path.join(data_path, "obstacles.csv"))
            color = obstacle_rgb
        else:
            # TODO error
            pass

        for line in position_data.itertuples(index = True, name ='Pandas'):
            if type_ == "box":
                self.boxes.append((getattr(line, "row"), getattr(line, "col")))
            elif type_ == "obstacles":
                self.obstacles.append((getattr(line, "row"), getattr(line, "col")))
            self.create_rectangle(
                x=getattr(line, "col") * self.unit,
                y=(height - getattr(line, "row") - 1) * self.unit,
                width=self.unit, 
                height=self.unit, 
                fill=color)

    def _add_rendering_entry(self, entry, permanent=False):
        if permanent:
            self.viewer.add_geom(entry)
        else:
            self.viewer.add_onetime(entry)
    
    def create_rectangle(self, x, y, width, height, fill):
        ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
        rect = FilledPolygon(ps)
        rect.set_color(fill[0], fill[1], fill[2])
        rect.add_attr(Transform())
        self._add_rendering_entry(rect, permanent=True)

    def create_circle(self, x, y, diameter, fill, resolution=20):
        c = (x + self.unit / 2, 
             y + self.unit / 2)
        dr = math.pi * 2 / resolution
        ps = []
        for i in range(resolution):
            x = c[0] + math.cos(i * dr) * diameter / 2
            y = c[1] + math.sin(i * dr) * diameter / 2
            ps.append((x, y))
        circ = FilledPolygon(ps)
        circ.set_color(fill[0], fill[1], fill[2])
        circ.add_attr(Transform())
        self._add_rendering_entry(circ, permanent=True)

    def remove_circle(self, x, y, diameter, resolution=20):
        if (x, y) in self.boxes:
            color = box_rgb
        elif (x, y) in self.obstacles:
            color = obstacle_rgb
        else:
            color = white_rgb
        c = (x + self.unit / 2, 
             y + self.unit / 2)
        dr = math.pi * 2 / resolution
        ps = []
        for i in range(resolution):
            x = c[0] + math.cos(i * dr) * diameter / 2
            y = c[1] + math.sin(i * dr) * diameter / 2
            ps.append((x, y))
        circ = FilledPolygon(ps)
        circ.set_color(color[0], color[1], color[2])
        circ.add_attr(Transform())
        self._add_rendering_entry(circ, permanent=True)

    def update_movement(self, new_pos, idx):
        self.create_circle(
            x = new_pos[1] * self.unit,
            y = (height - new_pos[0] - 1) * self.unit,
            diameter = self.unit / 2,
            fill=agent_rgb)
        self.movement.append(self.viewer.render(return_rgb_array=1))
        ##self.save_image(idx)
        self.remove_circle(
            x = new_pos[1] * self.unit,
            y = (height - new_pos[0] - 1) * self.unit,
            diameter = self.unit / 2)
        ##print(len(self.movement))
    def save_image(self, idx):
        render_arr = self.viewer.render(return_rgb_array=1)
        cv2.imshow('image', render_arr)
        cv2.imwrite(os.path.join(self.log_path, f"result_{idx}.png"), render_arr)

    def save_gif(self,epi):
        imageio.mimsave(
            os.path.join(self.log_path, './result'+str(int(epi))+'.gif'),
            np.array(self.movement))


if __name__ == "__main__":
    height = 10 # 10 * 10
    width = 9 # 9 * 10

    # start display to show image (internal)
    display = Display(visible=False, size=(width, height))
    display.start()
    import pyglet
    from pyglet.gl import *

    start_point = (9, 4)
    unit = 50
    screen_height = height * unit
    screen_width = width * unit
    log_path = "./logs"
    data_path = "./data"

    actions = [(8, 4), (7, 4), (7, 3), (7, 2), (7,1), (6,1), (5,1), (4,1), (3,1), (2,1),
              (2,2),(2,3),(2,4),(2,5),(2,6),(2,7), (3,7),(4,7),(5,7),(6,7),(7,7),(7,6),
              (7,5),(7,4),(8,4), (9,4)]
    render_cls = Render(screen_width, screen_height, unit, start_point, data_path, log_path)
    for idx, new_pos in enumerate(actions):
        render_cls.update_movement(new_pos, idx+1)

    render_cls.save_gif()
    render_cls.viewer.close()
