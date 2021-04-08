import vk_api, random, requests
from dfgo import dbinsert, dboutput
from conf import TOKEN
#импорт

vk_session = vk_api.VkApi(token=TOKEN)
session_api = vk_session.get_api()
def send(message, peer_id):
    vk_session.method('messages.send', {'peer_id': peer_id, 'message': message, 'random_id': random.randint(-2147483648, +214783648)})

def send_subscribes(mess="Fox проехала"):
    for id in dboutput('subscribes'):
        send(mess, id['id'])
