import vk_api, random, requests
from dfgo import dbinsert, dboutput
from keyboard import create_key #keyboard(count, text, color='default', one_time=False, inline=False), key()
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from conf import TOKEN
#импорт

my_id = 109460407

vk_session = vk_api.VkApi(token=TOKEN)
session_api = vk_session.get_api()
longpoll = VkBotLongPoll(vk_session, 203734150)
power = 1

def keyboard():
    key = open("key.js", "r+", encoding="Windows-1251").read()
    return key

print("Бот(3) запущен") #Пишем в консоль, чтобы знать, что проблем с авторизацией не было.
while True: #начало цикла
    if True: #Когда-то здесь был try:
	for event in longpoll.listen(): #прослушиваем все сообщени

	    if event.type == VkBotEventType.MESSAGE_NEW:

		#Объявление динамических переменных
		mess = event.object.message['text']
		peer_id = event.object.message['from_id']

		def send(message=None, peer_id=peer_id):
		    vk_session.method('messages.send', {'peer_id': peer_id, 'message': message, 'random_id': random.randint(-2147483648, +214783648)})

		def ksend(key, message, peer_id=peer_id):
		    vk_session.method("messages.send", { "peer_id": peer_id, "message": message, "random_id": random.randint(1, 2147483647), "keyboard": key})

		def subscribe_send(peer_id=peer_id):
		    create_key([1], ['Подписаться'], ['primary'], True)
		    key=keyboard()
		    ksend(key, "&#128521;", peer_id)

		def subscribe(id=peer_id):
		    dbinsert("subscribes", str(id))
		    send('Вы подписались на уведомления о появлении Лисы', id)

		def send_subscribes(mess="Fox проехала"):
		    for id in dboutput('subscribes'):
			send(mess, id)

		if mess == "Начать":
		    subscribe_send(peer_id)

		if mess == "Подписаться":
		    subscribe(peer_id)

		if mess == "23032017":
		    power = 0
