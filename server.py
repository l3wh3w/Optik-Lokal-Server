from aiohttp import web
import datetime
from OMR_Main import process
from answer_key import saveAnswerKey


async def sendImage(request):
    data = await request.post()
    image = data.get("image")
    sinavKodu = data.get("sinavKodu")
    img_content = image.file.read()
    time = str(datetime.datetime.now().timestamp()).replace('.', '')
    with open(f'C:/Users/Mehmet/Desktop/local_server/photos/{time}.jpg', 'wb') as handler:
        handler.write(img_content)
    response = process(time, sinavKodu)
    return web.json_response(response)


async def AnswerKey (request):
    data = await request.post()
    image = data.get("image")
    sinavKodu = data.get("sinavKodu")
    img_content = image.file.read()
    time = str(datetime.datetime.now().timestamp()).replace('.', '')
    with open(f'C:/Users/Mehmet/Desktop/local_server/answer_key/{time}.jpg', 'wb') as handler:
        handler.write(img_content) 
    response = saveAnswerKey(time, sinavKodu)
    return web.json_response(response) 


app = web.Application()
app.add_routes([
    web.post('/sendImage', sendImage),
    web.post('/answerKey', AnswerKey),  # Add a new route for AnswerKey
])

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0')
