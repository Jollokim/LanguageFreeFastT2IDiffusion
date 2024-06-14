import numpy as np
from numpy import dot
from numpy.linalg import norm



def get_angle(p_level=0.5):
    std = 1

    radians = np.arcsin(std*p_level)
    angle = radians * (180/np.pi)

    print(angle)

def get_p_level(angle=30):
    std = 1
    
    radians = angle * (np.pi/180)
    p_level = np.sin(radians/std)

    print(p_level)



def main():
    get_angle(1)
    get_p_level(1)


if __name__ == '__main__':
    main()

    

