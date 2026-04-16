import cv2
import numpy as np
from ultralytics import YOLO
import torch
model = YOLO('yolo26n-seg.pt')
#completely random video 
video_path = r"D:\Video\15359923_1920_1080_120fps.mp4"
video = cv2.VideoCapture(video_path)

#makes sure the path is right
if not video.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

gallery_canvas = np.zeros((1080, 1920, 3), dtype=np.uint8) #for the pngs
currentX, currentY = 20, 20 #initializing the png printing 
row_height = 0  
track_history = {} #id of all cars 
archived_id = set() #id of cars are stored here after 30 frames as set in if loop below
standard_w = 300  #width of mask
standard_h = 200  #height of the mask
point_x = 971 # x point after which you want the car to be masked
point_y = 489 # y point after which you want the car to be masked

while True:
    status, image = video.read()
    
    if not status:
        break
    image= cv2.resize(image,(1920,1080))
    results = model.track(image, persist=True, conf=0.5) #prediction using yolo26l-seg model the track is use to provide unique id to each car
    classes = results[0].names 

    if results[0].masks is not None and results[0].boxes.id is not None: #makes sure that mask is there is a mask before we proceed
        masks = results[0].masks.data
        boxes = results[0].boxes.xyxy
        clss = results[0].boxes.cls
        track_ids = results[0].boxes.id.int().cpu().tolist() #inputs all the ids to the track_ids
    
        for i in range(len(masks)): #loop for every mask 
            if classes[int(clss[i])] == "car":  #makes sure we only print cars
                track_id = track_ids[i]   #sets the current cars id as track_id
                x1, y1, x2, y2 = map(int, boxes[i]) #cordinates of the box
                label = classes[int(clss[i])] # class
                if y1>point_y and track_id not in archived_id: 
                    '''counts for every frame of a car and only takes a mask
                     after 30 frames to make sure that the car is visible as a whole and not cut by the border'''                                                                 
                    archived_id.add(track_id) #then finally adds the car into archieved_id after 30 frames
                    raw_mask = masks[i]
                    resized_mask = cv2.resize(raw_mask.cpu().numpy(), (image.shape[1], image.shape[0])) #resize to the size of the image to make sure we can overlap the two
                    mask_3d = np.repeat(resized_mask[:, :, np.newaxis], 3, axis=2) #the masks pred are gray scale but we want it in 3 color channel that is BGR so we conver it to a 3d layer with BGR values for each
                   
                    isolated_object = (image * mask_3d).astype(np.uint8) #finally cuts out the final mask form original image
                    y1, y2 = np.clip([y1, y2], 0, image.shape[0])  #it safeguards the negative values
                    x1, x2 = np.clip([x1, x2], 0, image.shape[1])  #so this just helps in keeping the values inside the image size

                    crop = isolated_object[y1:y2, x1:x2] #crops the mask to box dimentions to make sure we can change its position
                    if crop.size == 0: #prevents crash if y1=y2 in between the frames
                        continue
                    mask_area_pixels = np.count_nonzero(crop > 0.5) #checks for pixels with value >0.5
                    total_crop_pixels = crop.size #total number of pixels in the mask
                    fill_percentage = (mask_area_pixels / total_crop_pixels) #%of mask there 
                    if fill_percentage < 0.5:
                        continue
                    f_final_car_crop = cv2.resize(crop, (standard_w, standard_h)) #crop the mask to the given dimentions in the start
                    h, w = standard_h, standard_w
                    if currentX + w > 1200: #this loop makes sure that rows change as they re filled up
                        currentX = 20
                        currentY += row_height + 20 #makes sure the distance between 2 rows is 20
                        row_height = 0
                    if currentY + h < 1060:
                        target_area = gallery_canvas[currentY:currentY + h, currentX:currentX + w] #this sets the target area where the mask goes
                        gallery_canvas[currentY:currentY + h, currentX:currentX + w] = \
                            np.where(f_final_car_crop > 0, f_final_car_crop, target_area) #this is the main part cause it overlays the mask to the canvas
                        
                        currentX += w + 20 #goes to next block to print next mask
                        row_height = max(row_height, h) #makes sure highest mask is considered before cropping
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1) #displays the box
                    cv2.putText(image, f"ID:{track_id} {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1) #displays the class
        
    final_img = np.concatenate((image, gallery_canvas), axis=1) #combine the original video and the canvas

    display_res = cv2.resize(final_img, (1920, 540))  #squishes down the 3840 of width due to concat to 1920 to make sure we see both the screens
    
    cv2.imshow("Detection Dashboard", display_res)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()