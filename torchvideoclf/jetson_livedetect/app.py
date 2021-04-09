from aiohttp import web
import socketio
import cv2
import threading
import time
import asyncio

connection_flag = 0

## creates a new Async Socket IO Server
sio = socketio.AsyncServer()
## Creates a new Aiohttp Web Application
app = web.Application()
# Binds our Socket.IO server to our Web App
## instance
sio.attach(app)

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.resize(frame, (480, 320))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

async def video_feed(request):
    response = web.StreamResponse()
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    await response.prepare(request)

    for frame in gen_frames():
        await asyncio.sleep(0.1)
        await response.write(frame)
    return response
## we can define aiohttp endpoints just as we normally
## would with no change
async def index(request):
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.on('connect')
async def connect_handler(sid, environ):
    global connection_flag
    print("new connection") # works as expected
    connection_flag = 1
    await sio.emit('initial_config', "connected") # works as expected

async def send_message():
    global connection_flag
    try:
        print("in send message")
        await asyncio.sleep(1)
        print("wait till a client connects")
        while connection_flag == 0:
            await asyncio.sleep(1)
            pass
        print("waiting 2 seconds..")
        await asyncio.sleep(2)
        i = 0
        while True:
            print("now emitting: ", i)
            # await sendData(i)
            await sio.emit('feedback', "From thread!")
            i += 1
            await asyncio.sleep(1)

    finally:
        print("finished, exiting now")
    # print("In send message!")
    # await socketio.sleep(2.0)
    # await sio.emit('feedback', "From thread!")
## If we wanted to create a new websocket endpoint,
## use this decorator, passing in the name of the
## event we wish to listen out for
@sio.on('message')
async def print_message(sid, message):
    print("Socket ID: " , sid)
    print(message)
    ## await a successful emit of our reversed message
    ## back to the client
    await sio.emit('message', message[::-1])

## We bind our aiohttp endpoint to our app
## router
app.router.add_get('/', index)
app.router.add_get('/videostream', video_feed)

## We kick off our server
if __name__ == '__main__':
    # t = threading.Thread(target=await send_message)
    # t.daemon = True
    # t.start()
    sio.start_background_task(target=lambda: send_message())
    web.run_app(app)

    cap.release()