# import xml.etree.ElementTree as ET

# mycar = ET.parse('/home/user/landing/g_vehicle/car_v3.urdf')
# root = mycar.getroot()

# for child in root:
#     if child.tag == 'material':
        
#         if child.get('name') == 'base_color':
#             child[0].attrib = {'rgba': '0.0 0.7 0.0 1.0'}



# mycar.write('/home/user/landing/g_vehicle/car_v3_parsed.urdf')


path = '/home/user/landing/g_vehicle/car_v3.urdf'
mycar = open(path, 'r')
lines = mycar.readlines()
mycar.close()


color = ['0.0', '1.0', '0.0', '1.0']   # ['0.0', '1.0', '0.0', '1.0']

for index, line in enumerate(lines):
    if line == '  <material name="base_color">\n':
        lines[index+1] = '    <color rgba="{} {} {} {}"/>\n'.format(color[0], color[1], color[2], color[3])

mycar = open(path, 'w')
mycar.writelines(lines)
mycar.close()
