# -*- coding: utf-8 -*- 
from vk_api.utils import get_random_id
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
#count=[] - количество кнопок в строке. [2, 3] - выдаст 2 кнопки в первой и 3 во второй строке.
#text[] - текст в каждой кнопке. Указывыается по очереди
#color[] - цвет каждой кнопки. Указывается по очереди

def in_key(count, text, color, one_time, inline):
	d = 0
	k = 0
	with open('key.js', 'w+') as egasdrhs:
		True
	with open('key.js', 'r+') as f:
		f.write(
'''{
  "one_time": ''' + str(one_time) + ''',
  "inline": ''' + str(inline) + ',')
		f.write('''
  "buttons": [''')
		for l in range(len(count)):
			d += k
			for i in range(count[l]):
						#print(i+d)
						if i == 0 and count[l] != 1:
							f.write(
'''							
    [{
      "action": {
	    "type": "text",
	    "label": "''' + text[i+d] + '''"
      },
      "color": "''' + color[i+d] + '''"
    },''')
						elif count[l] == 1 and l != (len(count) - 1):
							f.write(
'''							
    [{
      "action": {
	    "type": "text",
	    "label": "''' + text[i+d] + '''"
      },
      "color": "''' + color[i+d] + '''"
    }],''')							
						elif count[l] == 1 and l == (len(count) - 1):
							f.write(
'''							
    [{
      "action": {
	    "type": "text",
	    "label": "''' + text[i+d] + '''"
      },
      "color": "''' + color[i+d] + '''"
    }]''')
						elif i == (count[l] - 1) and l != (len(count) - 1) and count[l] != 1:
							f.write(
'''
    {
      "action": {
	    "type": "text",
	    "label": "''' + text[i+d] + '''"
      },
      "color": "''' + color[i+d] + '''"
      }],''')
						elif i == (count[l] - 1) and l == (len(count) - 1) and count[l] != 1:
							f.write(
'''
    {
      "action": {
	    "type": "text",
	    "label": "''' + text[i+d] + '''"
      },
      "color": "''' + color[i+d] + '''"
    }]''')
						else:
							f.write(
'''
    {
      "action": {
	    "type": "text",
	    "label": "''' + text[i+d] + '''"
      },
      "color": "''' + color[i+d] + '''"
    },''')
						k = i + 1
		f.write('''
  ]
}''')
def create_key(count, text, color='default', one_time=False, inline=False):
	for i in count:
		i+=i
	if color == 'default':
		color = ['default']*i
	one_time = str(one_time).lower()
	inline = str(inline).lower()
	in_key(count, text, color, one_time, inline)
def key():
	key = str(open("key.js", "r+", encoding="Windows-1251").read())
	return key
	

	