import vk_api, random, requests
from dfgo import dbinsert, dboutput
#импорт

vk_session = vk_api.VkApi(token="93bb564df2600c56bc902a86c6a130ded07afb7eac1eb9772c200b9215ea894beafa0a18ecefcbbc49b5e")
session_api = vk_session.get_api()
def send(message, peer_id):
    vk_session.method('messages.send', {'peer_id': peer_id, 'message': message, 'random_id': random.randint(-2147483648, +214783648)})

def send_subscribes(mess="Fox проехала"):
    for id in dboutput('subscribes'):
        send(mess, id['id'])

print("Бот(3) запущен") #Пишем в консоль, чтобы знать, что проблем с авторизацией не было.