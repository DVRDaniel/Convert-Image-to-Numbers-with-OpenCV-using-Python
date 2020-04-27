# Convert image table into a list of numbers in a text file for each column of the table

# Topics used
#   lists (pop, sorted)
#   np.arrays (max, delete)
#   files (opening, writing, closing)
#   threshold
#   contours (hierarchy, area, permiter, centroid)

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image with numbers in a table
covidData = cv2.imread("GtoCovid17Apr.png") # Load the data image
covidDataCopy = covidData.copy()            # Size of the image: 1080,756,3
# Output files to save the numbers of the table
investigacion = open("investigacion.txt", "w")
confirmados = open("confirmados.txt", "w")
recuperados = open("recuperados.txt", "w")
defunciones = open("defunciones.txt", "w")
comunitaria = open("comunitaria.txt", "w")
# Size of the table
nRows = 47   # 47 rows in table, total not included
nCols = 5    # 5 columns in table
# Size of cells and starting point os table
yCellSize = 21.7916
y0 =34

xCellSize = 104
x0 = 243

digits = 10     # number of guessing digits
# Examine all the tables 1row and all columns
for i in range(nRows):                              
    for j in range(nCols):
# Take one cell at a time and convert it to have a white number (s) and black background
        row = y0 + i * yCellSize
        col = x0 + j * xCellSize
        dataRoi = covidData[round(row) : round(row+yCellSize), col : col + xCellSize]
        grayCell = cv2.cvtColor(dataRoi, cv2.COLOR_BGR2GRAY)
        retval, whiteN_cell = cv2.threshold(grayCell, 50, 255, cv2.THRESH_BINARY_INV)
        nCellHeight, nCellWidth = whiteN_cell.shape
# Find the contours with tree-hierarchy info to later choose only the outer parts of the numbers
# RETR_EXTERNAL nos used because could be noise inside the cell
        contours_cell, hierarchyCell = cv2.findContours(whiteN_cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# Remove contours of low area (noise contours)
        nContCell = len(contours_cell)                  
        zerosAt = np.zeros((nContCell, ), dtype = int)  
        n_zeros = 0
        for index, cont in enumerate(contours_cell):    # Count number of contours with low Area
            if cv2.contourArea(cont) < 5:               #>   <
                zerosAt[n_zeros] = index - n_zeros
                n_zeros = n_zeros + 1
        for index in range(n_zeros):                    # remove elements with a despreciable area
            contours_cell.pop(zerosAt[index])           # when pop is executed a dim substracted
            hierarchyCell = np.delete(hierarchyCell, zerosAt[index], 1) # Not possible to pop from an array, need to delete
        nContCell = len(contours_cell)                  # Real number of contours 
# Detect number of childs, i.e. 8 has two childs, 0 one and 1 none
        nChilds = 0
        childCont = [0] * nContCell
        for index, cont in enumerate(contours_cell):
            if (hierarchyCell[0, index, 3]) != -1:      # If contour has a parent(column3) mean is a child
                childCont[nChilds] = index
                nChilds = nChilds + 1
        for index in range(nContCell - nChilds):
            childCont.pop()                             # Pop all zeros in childCont
# Number of digits in the cell
        cellDigits = nContCell - len(childCont)
        rect = [np.zeros(4, dtype = int)] * cellDigits    #x,y,w,h
        iDigit = 0
        for index, cont in enumerate(contours_cell):
            if not(index in childCont):                 # Verify that the actual contour is not a child
                rect[iDigit] = cv2.boundingRect(cont)   # Tuple modified at once, not element by element
                iDigit = iDigit + 1                     # Order is working good but two same rectangles are given,
        rectSort = sorted(rect, key=lambda x: x[0])     # need to redifine child or parent column of hierarchyCell   
                            # WARNING if fail, redifine hierarchyCell child and parent because contours were POP
# Examine one digit at a time inside the cell, a patch need to be set in all the remaining digits
        for iCellDigit in range(cellDigits):            # number of digits in the cell
            whiteNcellModf = whiteN_cell.copy()
            for iDigit in range(cellDigits):            
                if iCellDigit != iDigit:                # Set a black patch in the digits you dont want to examine
                    whiteNcellModf[rectSort[iDigit][1]:rectSort[iDigit][1]+rectSort[iDigit][3], rectSort[iDigit][0]:rectSort[iDigit][0]+rectSort[iDigit][2]] = 0
            #plt.figure(figsize = [5, 5])
            #plt.subplot(121), plt.imshow(whiteNcellModf), plt.title('whiteNcellModf')

# Cell with just one digit    
            contours_cell, hierarchy = cv2.findContours(whiteNcellModf, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
# 1 Erase Zero Area contours and get Areas, Perimeters and the centroid of the contour with the maxArea
            nContCell = len(contours_cell)                  
            zerosAt = np.zeros((nContCell, ), dtype = int)  
            n_zeros = 0
            for index, cont in enumerate(contours_cell):    # Count number of contours with low Area
                if cv2.contourArea(cont) < 5:               #>   <
                    zerosAt[n_zeros] = index - n_zeros
                    n_zeros = n_zeros + 1

            for index in range(n_zeros):                    # remove elements with a despreciable area
                contours_cell.pop(zerosAt[index])           # when pop is executed a dim substracted
            nContCell = len(contours_cell)
            
            nAreas = [(cv2.contourArea(cont), cont) for cont in contours_cell]
            maxArea = max(nAreas, key=lambda x: x[0])[0]    # Find largest contour, his center will be used to do imgAND
            index_of_maxArea = -1
            for index, cont in enumerate(contours_cell):    # Find the index of the largest area
                if cv2.contourArea(cont) == maxArea:
                    index_of_maxArea = index
                    break
            
            cellNarea = np.zeros((nContCell, ), dtype =  np.float64)
            cellNperim = np.zeros((nContCell, ), dtype =  np.float64)
            for index, cont in enumerate(contours_cell): 
                cellNarea[index] = cv2.contourArea(cont)
                cellNperim[index] = cv2.arcLength(cont, True)
                if index_of_maxArea == index:
                    M = cv2.moments(cont)               # Contour moments to find centroid
                    xCellN = int(round(M["m10"]/M["m00"]))
                    yCellN = int(round(M["m01"]/M["m00"]))  
            #print("Areas = {}".format(cellNarea))
            #print("Perims = {}".format(cellNperim))
        # 1 End
            minAdiff = 50   # Big area and Perim set so that enter 1st time when trying to get the contour with the minAdiff
            minPdiff = 10
# Iterate through all the numbers to identify which number is in the cell
            for k in range(digits):             # Guess the number
# Load a different number image each iteration
                if k == 0: number = cv2.imread("Numbers/0.jpg")
                elif k == 1: number = cv2.imread("Numbers/1.jpg")
                elif k == 2: number = cv2.imread("Numbers/2.jpg")
                elif k == 3: number = cv2.imread("Numbers/3.jpg")
                elif k == 4: number = cv2.imread("Numbers/4.jpg")
                elif k == 5: number = cv2.imread("Numbers/5.jpg")
                elif k == 6: number = cv2.imread("Numbers/6.jpg")
                elif k == 7: number = cv2.imread("Numbers/7.jpg")
                elif k == 8: number = cv2.imread("Numbers/8.jpg")
                else: number = cv2.imread("Numbers/9.jpg")
                #print("Guess = {}".format(k))
# Convert the number to white and black background
                grayNumber = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
                retval, whiteN = cv2.threshold(grayNumber, 123, 255, cv2.THRESH_BINARY_INV)
                nHeight, nWidth = whiteN.shape
                numberCopyCont = number.copy()      # Copy of number and cell to draw contrours and centroids
                contoursN, hierarchy = cv2.findContours(whiteN, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Skip the guessing number if the number in the cell has different amount of contours
                nContN = len(contoursN)
                if nContN != nContCell:
                    #print("{} cont dont match".format(k))
                    continue    # Try with the next digit because after comparing nContours not match
# Get Areas, Perimeters of all contours and the Centroid of the maxArea contour
                nAreas = [(cv2.contourArea(cont), cont) for cont in contoursN]
                maxArea = max(nAreas, key=lambda x: x[0])[0]    # Find largest contour, his center will be used to do imgOR
                index_of_maxArea = -1
                for index, cont in enumerate(contoursN):  # Find the index of the largest area
                    if cv2.contourArea(cont) == maxArea:
                        index_of_maxArea = index
                        break
                areaN = np.zeros((nContN, ), dtype =  np.float64)
                perimN = np.zeros((nContN, ), dtype =  np.float64)
                for index, cont in enumerate(contoursN):    # Contours in the guessing number
                    areaN[index] = cv2.contourArea(cont)
                    perimN[index] = cv2.arcLength(cont, True)
                    if index_of_maxArea == index:
                        M = cv2.moments(cont)               # Contour moments to find centroid
                        xN = int(round(M["m10"]/M["m00"]))
                        yN = int(round(M["m01"]/M["m00"]))

# Skip the guessing number if it's perimeter is far away from the cellNumber
                guessed = 1
                for p in range(nContCell):  #Helped to detect 5 and 7 correctly
                    #print("cellNperim {:.2f}, perimN {:.2f}".format(cellNperim[p], perimN[p]))
                    perimDiff = np.abs(cellNperim[p] - perimN[p])
                    if perimDiff > 7.2:
                        guessed = 0
                        break
                if guessed == 0:
                    #print("PerimDiff of cellN with guessingN too far away")
                    continue    # Big diff on perim, try with the next digit
            
# Create a black cell with the centroid of guessingNumber matching the centroid of cellNumber
                maskCell = np.zeros((nCellHeight, nCellWidth), dtype = np.uint8)
                row0 = yCellN - yN
                rowf = yCellN + nHeight - yN
                col0 = xCellN - xN
                colf = xCellN + nWidth - xN
                whiteNadj = whiteN.copy()
# Skip the guessing number if the centroids have a big offset
                if row0 < 0 or rowf > nCellHeight:
                    #print("Offset of centroids")
                    continue    # Try with the next digit because offset in centroids
# Overlap the guessing number in the cell number and get the it's contours
                maskCell[row0:rowf, col0:colf] = whiteNadj
                CellVSn = cv2.bitwise_or(whiteNcellModf, maskCell)        # OR op to guessing number and dataCell
                #plt.subplot(122), plt.imshow(CellVSn), plt.title('CellVSn')
                #plt.show()
                contoursComp, hierarchy = cv2.findContours(CellVSn, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Delete zero area contours (noise)
            # Like 1
                NcontComp = len(contoursComp)
                zerosAt = np.zeros((NcontComp, ), dtype = int)  
                n_zeros = 0
                for index, cont in enumerate(contoursComp):    # Count number of contours with low Area
                    if cv2.contourArea(cont) < 5:               #>   <
                        zerosAt[n_zeros] = index - n_zeros
                        n_zeros = n_zeros + 1

                for index in range(n_zeros):                    # remove elements with a despreciable area
                    contoursComp.pop(zerosAt[index])           # when pop is executed a dim substracted
                NcontComp = len(contoursComp)
                if NcontComp != nContCell:
                    #print("Number of cont NOT match")
                    continue    # Try with the next digit because after comparing nContours not match
# Get the Area, Perimeters of all contours and the centroid of the maxArea contour          
                nAreas = [(cv2.contourArea(cont), cont) for cont in contoursComp]
                maxArea = max(nAreas, key=lambda x: x[0])[0]    # Find largest contour, his center will be used to do imgAND
                index_of_maxArea = -1
                for index, cont in enumerate(contoursComp):  # Find the index of the largest area
                    if cv2.contourArea(cont) == maxArea:
                        index_of_maxArea = index
                        break
                areaComp = np.zeros((NcontComp, ), dtype =  np.float64)
                perimComp = np.zeros((NcontComp, ), dtype =  np.float64)        
                for index, cont in enumerate(contoursComp): 
                    areaComp[index] = cv2.contourArea(cont)
                    perimComp[index] = cv2.arcLength(cont, True)
                    if index_of_maxArea == index:
                        M = cv2.moments(cont)   # Contour moments to find centroid
                        xComp = int(round(M["m10"]/M["m00"]))
                        yComp = int(round(M["m01"]/M["m00"]))
            # End Like 1          
                # All passes in this cond....Before being rejected by area compare centroids (xCellN,yCellN) vs (xComp, yComp)
                # if xCellN == xComp and np.abs(yCellN - yComp) < 2:
                    # print("At {}, {} was found {} centroids".format(i, j, k))
                    # break
# Skip the guessing number if the area and perimeter have a big difference
                # Compare cellNarea with areaComp, if diff is higher than 18 skip
                # Compare cellNperim with perimComp, if diff is higher than 7.2 skip
                for p in range(nContCell):
                    area_limit = 19
                    perim_limit = 7.2
                    areaDiff = np.abs(cellNarea[p] - areaComp[p])
                    perimDiff = np.abs(cellNperim[p] - perimComp[p])
                    if areaDiff > area_limit or perimDiff > perim_limit:
                        if areaDiff < minAdiff and perimDiff < minPdiff:    # Helped to detect 2 but still fail at 5 and 7
                            minAdiff = areaDiff
                            minPdiff = perimDiff
                            guess = k
                        guessed = 0
                        break
                if guessed == 0:
                    #print("Big diff on AreaComp with cellN")
                    continue    # Big diff on area, try with the next digit
                
# If passed all the locks means is the guessing number
# Show result in screen and store in text files, write exact guesses
                if iCellDigit == 0 and cellDigits > 1:
                    print("At {}, {} was found           {}".format(i, j, k), end = '')
                    if(j == 0): investigacion.write("{}".format(k))
                    if(j == 1): confirmados.write("{}".format(k))
                    if(j == 2): recuperados.write("{}".format(k))
                    if(j == 3): defunciones.write("{}".format(k))
                    if(j == 4): comunitaria.write("{}".format(k))
                elif iCellDigit == 0:
                    print("At {}, {} was found           {}".format(i, j, k))
                    if(j == 0): investigacion.write("{}\n".format(k))
                    if(j == 1): confirmados.write("{}\n".format(k))
                    if(j == 2): recuperados.write("{}\n".format(k))
                    if(j == 3): defunciones.write("{}\n".format(k))
                    if(j == 4): comunitaria.write("{}\n".format(k))
                elif iCellDigit == cellDigits -1:
                    print("{}".format(k))
                    if(j == 0): investigacion.write("{}\n".format(k))
                    if(j == 1): confirmados.write("{}\n".format(k))
                    if(j == 2): recuperados.write("{}\n".format(k))
                    if(j == 3): defunciones.write("{}\n".format(k))
                    if(j == 4): comunitaria.write("{}\n".format(k))
                else:
                    print("{}".format(k), end = '')
                    if(j == 0): investigacion.write("{}".format(k))
                    if(j == 1): confirmados.write("{}".format(k))
                    if(j == 2): recuperados.write("{}".format(k))
                    if(j == 3): defunciones.write("{}".format(k))
                    if(j == 4): comunitaria.write("{}".format(k))
                break                                                   # End of for of 10 numbers guessing
# Guess number selected based on the lowest diff in area and perimeter
            if guessed == 0:
                # print("At {}, {} maybe is            {}".format(i, j, guess))
                if iCellDigit == 0 and cellDigits > 1:
                    print("At {}, {} was found           {}".format(i, j, guess), end = '')
                    if(j == 0): investigacion.write("{}".format(guess))
                    if(j == 1): confirmados.write("{}".format(guess))
                    if(j == 2): recuperados.write("{}".format(guess))
                    if(j == 3): defunciones.write("{}".format(guess))
                    if(j == 4): comunitaria.write("{}".format(guess))
                elif iCellDigit == 0:
                    print("At {}, {} was found           {}".format(i, j, guess))
                    if(j == 0): investigacion.write("{}\n".format(guess))
                    if(j == 1): confirmados.write("{}\n".format(guess))
                    if(j == 2): recuperados.write("{}\n".format(guess))
                    if(j == 3): defunciones.write("{}\n".format(guess))
                    if(j == 4): comunitaria.write("{}\n".format(guess))
                elif iCellDigit == cellDigits -1:
                    print("{}".format(guess))
                    if(j == 0): investigacion.write("{}\n".format(guess))
                    if(j == 1): confirmados.write("{}\n".format(guess))
                    if(j == 2): recuperados.write("{}\n".format(guess))
                    if(j == 3): defunciones.write("{}\n".format(guess))
                    if(j == 4): comunitaria.write("{}\n".format(guess))
                else:
                    print("{}".format(guess), end = '')
                    if(j == 0): investigacion.write("{}".format(guess))
                    if(j == 1): confirmados.write("{}".format(guess))
                    if(j == 2): recuperados.write("{}".format(guess))
                    if(j == 3): defunciones.write("{}".format(guess))
                    if(j == 4): comunitaria.write("{}".format(guess))
# Close files with numbers from table in the image
investigacion.close()
confirmados.close()
recuperados.close()
defunciones.close()
comunitaria.close()
