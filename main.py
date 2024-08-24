import os
import cv2
import numpy as np
import sys
from collections import defaultdict, deque 
from morphological import morphOpen1, morphOpen2
from erode import erosion
from gaussian import gaussianblur
from dilate import dilation
from addweighted import addWeighted
from checkVisited import checkvisitedOrNot
from mathss import findAngelMean, findAngelVariance, findBinary,  findCentroid, findAngle
import pandas as pd


dir = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]





def findSutures(imgx):
  img = cv2.imread(imgx)
  details = []

  grey= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  angles = []
  guassian= gaussianblur(grey, 5, verbose=False)
  sharp=addWeighted(grey, 1.5,guassian,-0.5,0)
  invert = findBinary(sharp, 127)
  firstimage = morphOpen1(invert)
  h = []
  w = []
  secondimage = morphOpen2(invert)
  bitand= np.bitwise_and(firstimage,secondimage)
  counter = -1

  n, m = bitand.shape
  differences = []
  centroids = []
  vis = [[0] * m for _ in range(n)]

  i, j = 0, 0
  mp = defaultdict(list)
  while i < n:
      j = 0
      while j < m:
          if bitand[i][j] == 255 and not vis[i][j]:
              vis[i][j] = 1  

              tempQueue = deque()
              counter += 1

              tempQueue.append((i, j)) 
              while tempQueue:
                  i, j = tempQueue.popleft()
                  mp[counter].append((i, j))          
                  for di, dj in dir:
                      ni, nj = i + di, j + dj
                      if checkvisitedOrNot(ni, nj, bitand, vis):
                          vis[ni][nj] = 1
                          tempQueue.append((ni, nj))
          j += 1
      i += 1
      


  # Compute component  
  component = []
  area = []
  for k,v in mp.items():
      
      
     
      
      
      ymin = float("inf")
      vari=float("inf")
      xmin=max(0,vari)
      greater=float("-inf")
      ymax= min(0,greater)
      ctr  =float("-inf")
      xmax=min(0,ctr)
      
      for x, y in v:
          xmax = x if x > xmax else xmax
          xmin = min(xmin, x)
          ymax = ymax if ymax > y else y
          ymin = min(ymin, y)
          
      diffofy=ymax - ymin
      diffofx=xmax - xmin
      combinexy=(ymax - ymin)*(xmax - xmin)
      component.append([ymin, xmax,ymax,xmin, diffofy, diffofx, combinexy])

      h.append(diffofy)
      area.append(combinexy)
      w.append(diffofx)

  
  a_AngelMean = np.mean(area) * 0.5

  # Compute details
  
  for k,v in mp.items():
      if area[k] > a_AngelMean:
        y1, y2, c1, c2 = 0, 0, 0, 0
        ymin, ymax = float("inf"), float("-inf")
        
        for x, y in v:
            ymax = y if y > ymax else ymax
            ymin = ymin if y > ymin else y
            
        for x, y in v:
            if y == ymax:
                y2 = x+ y2
                c2 += 1
                
            if y == ymin:
                y1 = x+y1 
                c1 += 1
                
                
        # details.append([y1//c1, ymin, y2//c2, ymax])
        details.append([ymin,y1//c1, ymax,y2//c2 ])
      
 
  for d in details:  
    cv2.circle(img, (d[0], d[1]), 5, (0, 255, 0), -1)
    angle = findAngle(d[0], d[1], d[2], d[3])
    angles.append(angle)
    cv2.circle(img, (d[2], d[3]), 5, (0, 255, 0), -1)
      


  # Compute centroids
  
  
  for d in details:  
    cx, cy = findCentroid(d[0], d[1], d[2], d[3])
    centroids.append((cx, cy))
    cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

  himg = img.shape[0]
  for i in range(1, len(centroids)):
    y_diff = centroids[i][1] - centroids[i - 1][1]
    differences.append(y_diff/himg)

  
  

  centeroidM = np.mean(differences)

  print(f"AngelMean: {centeroidM}")

  AngelMean = findAngelMean(angles)
  print(f"Angle AngelMean: {AngelMean}")

  centroidV = np.var(differences)
  print(f"AngelVariance: {centroidV}")
  # Calculate Angelvariance

  Angelvariance = findAngelVariance(angles, AngelMean)
  print(f" Angkle AngelVariance: {Angelvariance}")

  cv2.imshow('connected components', img)

  if(cv2.waitKey(0)==27):
      cv2.destroyAllWindows()
  mp = defaultdict(list)
  return len(centroids), centeroidM, centroidV, AngelMean, Angelvariance


def process_All_images(input_dir, output_csv):
    data = []
    # Get a list of all files in the input directory
    all_files = os.listdir(input_dir)

    # Initialize index
    index = 0

    # Process files until all files have been examined
    while index < len(all_files):
        image_name = all_files[index]

        # Check if the file is a PNG image
        if image_name.endswith('.png'):
            image_path = os.path.join(input_dir, image_name)
            num_sutures, AngelMean_inter_suture_spacing, Angelvariance_inter_suture_spacing, AngelMean_suture_angle, Angelvariance_suture_angle = findSutures(image_path)
            data.append([image_name, num_sutures, AngelMean_inter_suture_spacing, Angelvariance_inter_suture_spacing, AngelMean_suture_angle, Angelvariance_suture_angle])

        # Move to the next file
        index += 1

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(data, columns=['image name', 'number of sutures', 'AngelMean inter suture spacing', 'Angelvariance of inter suture spacing', 'AngelMean suture angle', 'Angelvariance of suture angle wrt x-axis'])
    df.to_csv(output_csv, index=False)

def compareImages(input_csv, output_csv):
    # Placeholder for actual comparison logic

# Read input CSV file
    df = pd.read_csv(input_csv)
    result_data = []

    # Initialize index
    index = 0

    # Process each row until all rows have been examined
    while index < len(df):
        row = df.iloc[index]
        img1_path, img2_path = row['img1 path'], row['img2 path']
        
        
        _, _, distVar1, _, angel_variance_suture_angle1 = findSutures(img1_path)
        output_distance = 2 if distVar1 > distVar2 else 1
        
        _, _, distVar2, _, angel_variance_suture_angle2 = findSutures(img2_path)
        output_angle = 2 if angel_variance_suture_angle1 > angel_variance_suture_angle2 else 1



        result_data.append([img1_path, img2_path, output_distance, output_angle])

        # Move to the next row
        index += 1

    # Create a DataFrame and save it to a CSV file
    result_df = pd.DataFrame(result_data, columns=['img1 path', 'img2 path', 'output distance', 'output angle'])
    result_df.to_csv(output_csv, index=False)




if __name__ == "__main__":
    output_csv = sys.argv[3]
    input_path = sys.argv[2]

    part_id = int(sys.argv[1])
    if part_id == 2:
        compareImages(input_path, output_csv)

    elif part_id == 1:
        process_All_images(input_path, output_csv)

    else:
        sys.exit(1)