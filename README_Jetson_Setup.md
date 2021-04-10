### Steps to setup Jetson for Action Recognition

The jetson xavier will be mounted on the dash of the car. The USB cam on the xavier will stream in the video feeds and the pre-trained model will predict if there is a 'pedestrian approaching' or a 'cyclist approaching' in the view of the camera. Note: this is different than just detecting if there is a pedestrian in the frame, while driving on the streets, there will be predestrians in the view, our approach here is to detect when the pedestrain is dangerously close to the vehicle or moving in a way that could potenially mean them intercepting the path of the moving vehicle. It's easy for humans to detect such situations since we have plethora of experiences detecting when a situation may develop with the slightest of the hints. Our attempt is to teach the model detect such situations. 
This model may have applications in self driving cars, however, when trained on other types of actions such as detecting the type of visitors walking in into a gated community and classifiying the objective of the visitor such as 'food delivery', 'guest', 'courier', 'housekeeping', 'residents' etc. helps automatic cataloging of entries that are today mostly manual or not present at all. This also has applications in the areas of retail where we can potentially detect customers picking up items from shelves, returning them to shelves, approaching the billing counter, window shopping, showing interest in products in a particular aisle etc.

In the current scope of the project where we are detecting 'clueless prdestrains' or 'cyclists', we have a few options on how to interact with the environment upon detection:

- Sound an alarm (can be done on the xavier itself, but we would need a buzzer etc.)
- Save the video clip where the action was detected
- Stream the live feed to an App on the phone or a web application and highlight when a potential action is detected


##### Detection from live feed

The frames recieved from the camera are buffered on the Jetson via a sliding window approach. Each frame initiates a new queue that keeps adding frames until a specified number of frames are collected. e.g. if we are going to do inference on a 3 second clip, assuming we are getting 15 fps from the usb cam on the jetson, we will have 45 frames in a queue, which will then be used for inference and then the frames will be discarded. Every second a new queue will be created, which means every 15 frames a new queue is created, at the end of 3 seconds we have 3 queues which the first queue having 45 frames, the 2nd one with 30 frames and the 3rd one with 15 frames. That is the maximum number of frames we will have in memory at a given time. As soon as a queue has 45 frames, we run the prediction and drop the frames.


##### Conecting to the Jetson on board a vehicle

Tried USB tethering, connected the Jetson to the iPhone via USB-A ports on the Jetson. Jetson is able to access the internet via the connection that is created, however, the phone cannot reach the Jetson since there is no IP available that is open to communication.

_When using USB tethering, the computer sees the smartphone as a USB modem. Your computer will not be getting an IP address as all outside communication is handled by the phone (modem).

_The only way the phone will have an IP address is if it is connecting via WiFi in which case, it would be simpler to connect your laptop to the same WiFi the phone is using instead._

_If the phone is using the cellular network, it will not have its own IP address either. It will be using its SIM ID on the cellular network instead. The IP address seen by other devices on the internet is a NATted address used by multiple devices on the cellular network._

Ideally, we sould use a M2 Wifi module USB based that can connect to the Jetson. This Wifi module will allow us to connect to the Jetson from the phone.

Or, we could use a Bluetooth module. Both these options would require procuring these gadgets which might take a while to arrive. 

A very common gadget available with most of us is the Wifi hotspot.

A 4G LTE hotspot device uses a SIM card to provide internet access to the devices connected to it. Both the Jetson and the phone will connect to thie hotspot device. This will assign local IP addresses to both the Jetson and the Phone. We can view these IPs from the management console of the hotspot. The App running on the phone can then connect to the Jetson using the IP address.

Since I do not have a hotspot, I would use the home Wifi router for this purpose assuming that it sould work the same way for the hotspot as well.

Another option is to use the Xavier's built in network device to create a sort of Wifi hotspot network that both the Jetson and the Phone can connect to then. This will allow the Phone to talk to the Jetson even though they will not have any connectivity to the internet.

Here are the steps to achieve this:

1. Ensure you have the serial cable plugged in to the Jetson and the screen tool is used to open a session to the Jetson.
2. Using the Ubuntu GUI (use VNC Viewer), create a new Wifi connection on the Xavier of mode = Hotspot

![image](https://user-images.githubusercontent.com/55649656/114140786-59426180-992e-11eb-9e47-511ad8ecc28d.png)

3. Connect the xavier to this new Wifi network
4. Connect your phone/laptop to this Wifi network

Both the phone and the xavier will get a 10.42.* address when on this network. You can use this IP to talk to the xavier from the phone.


#### Stream video from Jetson to a web browser
Read the frame from the video feed of the camera
encode the image in JPEG format
convert the encoded image to bytearray format
send the bytearray as image/jpeg content type in the http response
Content-Type: image/jpeg
bytearray(jpeg_encoded_image)

yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


get the docker image to use on the Jetson, and run the container with access to the camera

sudo docker run -it --rm --runtime=nvidia --device=/dev/video0 -v ~/w251/finalproject/app:/app  -p 8888:8888 mayukhd/jetson_4_1:cp36torch1.7


