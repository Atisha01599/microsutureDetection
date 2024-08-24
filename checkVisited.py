def checkvisitedOrNot(i, j, images, visitTrack):

    cols, rows =  len(images[0]), len(images)

    if 0 <= j < cols and 0 <= i < rows:
        # Check if the pixel value is 255 and has not been visitTrackited
        return images[i][j] == 255 and not visitTrack[i][j]

    return False

