import numpy as np
from math import pi,ceil
from cmath import rect,phase
import pygame
from time import sleep

red = (254, 0, 0)
white = (0, 0, 0)
green=(0,128,0)


def carr(er,car,phase0):
    x1=to_px(er,10+11*1j,(800,600))
    x2=to_px(car.position,10+11*1j,(800,600))
    x=x2[0]-x1[0]
    y=x2[1]-x1[1]
    mapii=(phase(point(x,y))/pi*180+180)
    if phase0>mapii and (abs(phase0-mapii)<100):
        return mapii,True
    return mapii,False

def neuron(car,score):
    '''car-класс содержащий параметры машины, позицию и направление
    Score-выход нейронной сети
    Return: новый объект car содержащий новое положение машинки'''
    if score == 0:
        car.direction=rect(abs(car.direction),phase(car.direction)+pi/6)
    elif score == 2:
        car.direction=rect(abs(car.direction),phase(car.direction)-pi/6)
    return car

def create_car():
    s = pygame.Surface((25, 15))
    s.set_colorkey(white)
    s.fill(white)
    pygame.draw.rect(s, red, pygame.Rect(0, 0, 15, 15))
    pygame.draw.polygon(s, red, [(15, 0), (25, 8), (15, 15)], 0)
    return s

def to_px(p, scale, size):
    """
    convert point from coordinate units to screen pixel units
    :param p: complex number representing point in coordinate units
    :param scale: complex number indicating how many coordinate units are from zero (center)
     to the right and to the top; zero is placed right in the center of the screen;
     ```scale=(1,1), size=(800,600)``` means that zero will be located at (400, 300) and
     ```p=-1j``` will be equal to pixel at (400, 600)
    :param size: int tuple indicating screen size ((800, 600), (1024,768), etc.)
    :return point: int tuple representing pixel ```p``` refers to
    """
    center = int(size[0] / 2)+ int(size[1] / 2)*1j
    unit = int(center.real / scale.real) + int(center.imag / scale.imag) * 1j
    return int(center.real + unit.real * p.real), int(size[1] - center.imag - unit.imag * p.imag)

def plot_map(mapi, screen, scale=None,sizeshow=(800,600), color=(0, 0, 0), width=2):
    """
    рисуем карту
    :param mapi: np.array complex point -информация о карте
    :param screen: холст на котором будем рисовать
    :param scale: complex point - начало рисования карты
    :param color: цвет фона
    :param with: толциа стен
    :return: complex digit - начало рисовки карты
    """
    if not scale:
        xmax, ymax = np.array([(abs(outer.real), abs(outer.imag)) for inner, outer in mapi]).max(axis=0)
        scale = ceil(xmax) + ceil(ymax) * 1j
    points = np.array([[to_px(inner, scale, sizeshow), to_px(outer, scale, sizeshow)] for inner, outer in mapi])
    pygame.draw.polygon(screen, color, points[:, 0], width)
    pygame.draw.polygon(screen, color, points[:, 1], width)
    return scale

def generate_map(sectors, radius, width, scale):
    """
    :param sectors: колличество секторов карты
    :param radius: average distance between 0 and inner point of map
    :param width: дистанция между внутрнними и внешними точками карты
    :param scale: scale of radius variation, as in np.random.normal(loc=radius, scale=scale, size=sectors)
    :return: list of tuples (`inner_point`, `outer_point`) of length :param sectors:
    """
    sector_angles = get_partition(sectors, -pi, pi)
    sector_radii = np.random.normal(loc=radius, scale=scale, size=sectors)
    sector_radii[sector_radii <= 0] = 1e-6
    inner_points = [rect(r, phi) for phi, r in zip(sector_angles, sector_radii)]
    outer_points = [rect(r, phi) for phi, r in zip(sector_angles, sector_radii + width)]
    return list(zip(inner_points, outer_points))

def get_agent_image( original, state, mapi,scale=None):# эти вещи используются для изображегияя машинки
    """используется для изображение машинки
    original-объект машикни
    state-класс содержащий информации о положении машинки и ее направления
    scale-комплексное число кордината карнты"""
    if not scale:
        xmax, ymax = np.array([(abs(outer.real), abs(outer.imag)) for inner, outer in mapi]).max(axis=0)
        scale = ceil(xmax) + ceil(ymax) * 1j
    angle = phase(state.direction) * 180 / pi
    rotated = pygame.transform.rotate(original, angle)
    rectangle = rotated.get_rect()
    rectangle.center = to_px(state.position, scale, (800,600))
    img=rotated, rectangle
    return img

class State:
    def __init__(self,pos,rect):
        self.position=pos
        self.direction=rect
    def copy(self):
        return State(self.position,self.direction)

def rotate(p, phi):
    """поворачивает вектор p на угол phi
    p-комплексное число
    phi-угол в радианах"""
    return rect(abs(p), phase(p) + phi)

def get_line_coefs(p1, p2):#нужна для нахождения пересечения
    """
    ситает коэфициенты линейного уравнения п двум комплексным точкам
    коэфициенты нормализованы, i.e. A + B + C = 1
    :param p1: комплексное число
    :param p2: комплексное число
    :return: numpy array of shape (3,) с коэфициентами A, B, C такие что
    A * p.real + B * p.imag + C = 0 для каждой точки из множества [p1, p2]
    """
    assert p1 != p2, "Line cannot be determined by one point! p1 = {:.5f} = {:.5f} = p2".format(p1, p2)
    for _ in range(10):
        try:
            a = np.array([[p1.real, p1.imag, 1], [p2.real, p2.imag, 1], [1, 1, 1]])
            b = [0, 0, 1]
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            p1 += p1 - p2
    raise np.linalg.LinAlgError("Cannot count line coefs for line determined by {:.5f} and {:.5f}".format(p1, p2))

def to_line_equation(coefs, p):
    """
    возврощвет значение линейного уравнения с значениями некой точки
    :param coefs: list коэфициентов уравнения прямой
    :param p: точка (комплексное число)
    :return: A * p.real + B * p.imag + C
    """
    A, B, C = coefs
    return A * p.real + B * p.imag + C


def intersect(l1, l2):
    """
    Находит точку пересечения двух прямых
    :param l1: list коэфициентов первой прямой [a1, b1, c1]
    :param l2: list коэфициентов второй прямой [a2, b2, c2]
    :return: точку ересечения двух прямых (комплексное число) a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0
    """
    from numpy.linalg import solve
    a = [l1[:2], l2[:2]]
    b = [-l1[-1], -l2[-1]]
    ans = solve(a, b)
    return ans[0] + ans[1] * 1j

def intersect_ray_with_segment(ray, segment):#проверяет на пересечение
    """
    :param ray: (ray_start, ray_direction) - tuple содержащий начало и напраление радара
    :param segment: (segment_start, segment_end) - tuple содержащий начало и конец прямой
    :return: complex point - содержит длину
    """
    r_start, r_dir = ray
    s_start, s_end = segment
    r_line = get_line_coefs(r_start, r_start + r_dir)
    s_line = get_line_coefs(s_start, s_end)
    intsct = intersect(r_line, s_line)
    if not (min(s_start.real, s_end.real) <= intsct.real <= max(s_start.real, s_end.real)):
        return None
    if not (min(s_start.imag, s_end.imag) <= intsct.imag <= max(s_start.imag, s_end.imag)):
        return None
    if r_dir.real * (intsct - r_start).real < 0 or r_dir.imag * (intsct - r_start).imag < 0:
        return None
    return intsct

def ray(original,mapi,car,num_layers,velocity=False,face=None,sizeshow=(800,600)):
    """
    param face: Холс на котором рисуем
    param mapi: np.array множества точек используемой карты
    param car: класс содержащий информацию о местоположении и направлении машинки
    param num_layers: колличество радаров
    return: длины радаров
    """
    xmax, ymax = np.array([(abs(outer.real), abs(outer.imag)) for inner, outer in mapi]).max(axis=0)
    scale = ceil(xmax) + ceil(ymax) * 1j
    raylist=list()
    startray=rect(1,phase(car.direction)-pi/2)
    alfa=pi/(num_layers-1)
    for i in range(num_layers):
        raylist.append(rect(1,phase(startray)+i*alfa))    
    sectors=len(mapi)
    otvet=[]
    layers=[]
    for j in raylist:
        mindir=10000+10000*1j
        for i in range(sectors):
            inner_wall = mapi[i - 1][0], mapi[i][0]
            outer_wall = mapi[i - 1][1], mapi[i][1]
            inter_inner=intersect_ray_with_segment((car.position,j),inner_wall)
            if inter_inner !=None:
                if abs(inter_inner-car.position)<abs(mindir-car.position):
                    mindir=inter_inner
            inter_outer=intersect_ray_with_segment((car.position,j),outer_wall)
            if inter_outer!=None:
                if abs(inter_outer-car.position)<abs(mindir-car.position):
                    mindir=inter_outer
        otvet.append(mindir)
    for i in otvet:
        start_pos = to_px(car.position, scale, sizeshow)
        end_pos=to_px(i,scale,sizeshow)
        if velocity:
            pygame.draw.line(face,green,start_pos,end_pos)
        layers.append(abs(car.position-i))
    if velocity:
        imgee=get_agent_image(original,car,mapi,scale)
        face.blit(imgee[0],imgee[1])
    return layers

def define_sector(mapi, position):#нужна для проверки столкновения с картой
    """
    param mapi: np.array- точки используемой карты
    param position: complex point- местоположения машинки
    """
    cur_phase = phase(mapi[-1][0]) - 2 * pi
    for i in range(len(mapi)):
        prev_phase = cur_phase
        cur_phase = phase(mapi[i][0])
        if min(prev_phase, cur_phase) < phase(position) <= max(prev_phase, cur_phase):
            # посиция не находится между обеими прямыми карты
            return i
    raise AssertionError("phase(%s) = %f was not found anywhere in the m" % (str(position), phase(position)))

def is_out_of_map(mapi, position):#проверяет столкновение с картой
       
        current_sector = define_sector(mapi, position)

        coefs = get_line_coefs(mapi[current_sector][0], mapi[current_sector - 1][0])
        sign_of_0 = to_line_equation(coefs, 0)
        sign_of_point = to_line_equation(coefs, position)
        if sign_of_0 * sign_of_point > 0:  # new point is on the same side of map's inner line as 0
            return True

        coefs = get_line_coefs(mapi[current_sector][1], mapi[current_sector - 1][1])
        sign_of_0 = to_line_equation(coefs, 0)
        sign_of_point = to_line_equation(coefs, position)
        if sign_of_0 * sign_of_point < 0:  # new point is on the other side of map's outer line than 0
            return True

        return False

def point(x, y):
    """
    преобразует x ,y - комплекное число и возвращает его
    """
    return x + y * 1j


def get_partition(n, a, b=None):
    """
    возврощает специальное множество случайных значений
    """
    import numpy as np
    if b is None:
        b = a
        a = 0
    sample = np.random.rand(n)
    return a + (b - a) * np.cumsum(sample / sample.sum())

