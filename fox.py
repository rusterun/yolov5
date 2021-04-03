import vk_api, random, requests
from dfgo import dbinsert, dboutput
from keyboard import create_key #keyboard(count, text, color='default', one_time=False, inline=False), key()
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
import config
#импорт

my_id = 109460407

vk_session = vk_api.VkApi(token="93bb564df2600c56bc902a86c6a130ded07afb7eac1eb9772c200b9215ea894beafa0a18ecefcbbc49b5e")
session_api = vk_session.get_api()
longpoll = VkBotLongPoll(vk_session, 203734150)
power = 1

#Блок статичных переменных(констант)
error = False
ex = ' '

def keyboard():
    key = open("key.js", "r+", encoding="Windows-1251").read()
    return key

print("Бот(3) запущен") #Пишем в консоль, чтобы знать, что проблем с авторизацией не было.
while True: #начало цикла
    if True:
        for event in longpoll.listen(): #прослушиваем все сообщени
            print(event)
		
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
                    print("S")
                    subscribe_send(peer_id)
                
                if mess == "Подписаться":
                    print("S")
                    subscribe(peer_id)
                
                if mess == "23032017":
                    print("S")
                    power = 0
            
            if error == True:
                send('Error: ' + ex)
                error = False

				

				

				

			
						
    '''except:
        ex = str(Exception)
        print(ex)
        error = True
        print("\n Error Time \n")'''