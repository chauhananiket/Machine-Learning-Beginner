
#importing xlwt library to write into xls
import xlwt
from decimal import *
from xlwt import Workbook

#create a workbook
wb = Workbook()
#create a sheet
sheet = wb.add_sheet('Sheet')

#percentages list
percentages = [23.6, 38.2, 50, 61.8, 78.6, 113, 123.6, 138.2, 161.8]
#add first row
for i in range(len(percentages)):
   sheet.write(0,i+1,str(percentages[i])+'%')

#user input
n = int(input('Enter number of elements: '))
#second row starts from index 1
row=1
a=1
print('Enter numbers: ')
for i in range(n):
    #User input
   val = float(input())
   #Add entered value to first column of the row
   sheet.write(row,0,str(val))
   #calculate each percentage
   for j in range(len(percentages)):
       result = (percentages[j]/100)*val
       #write result to sheet by rounding upto 3 decimal points
       sheet.write(row,j+1,str(a.quantize((int)result)))
   #increment the row by 1
   row+=1
#save Workbook as xls`
wb.save('percentages.xls')