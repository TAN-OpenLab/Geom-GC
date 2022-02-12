import pickle
import random
from cryptography.fernet import Fernet
import types
import os
import math
import numpy as np

import base64
#coding=utf-8
from struct import pack,unpack
import hashlib
from Crypto.Cipher import AES
import sys


BLOCK_SIZE = 16  # Bytes
pad = lambda s: s + (BLOCK_SIZE - len(s) % BLOCK_SIZE) * \
                chr(BLOCK_SIZE - len(s) % BLOCK_SIZE)
unpad = lambda s: s[:-ord(s[len(s) - 1:])]


def aesEncrypt(key, data):

    data=base64.b64encode(data)
    data = data.decode('utf8')
    data = pad(data)
    key=base64.b64encode(key)
    key = key.decode('utf8') 
    key = pad(key)
    key = key.encode('utf8')
    cipher = AES.new(key, AES.MODE_ECB)
    result = cipher.encrypt(data.encode())
    return result
    
def prf(byte1, byte2):   
    d = aesEncrypt(byte1, byte2)
    d1 = d[0:4]
    d2 = d[4:8]
    p1 = unpack('i',d1)
    p2 = unpack('i',d2)
    p1=float(p1[0])
    p2=float(p2[0])
    return p1, p2

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
def getCircle(p1, p2, p3):
    x21 = p2.x - p1.x
    y21 = p2.y - p1.y
    x32 = p3.x - p2.x
    y32 = p3.y - p2.y
    # three colinear
    if (x21 * y32 - x32 * y21 == 0):
        print ("Colinear!")
        return None
    xy21 = p2.x * p2.x - p1.x * p1.x + p2.y * p2.y - p1.y * p1.y
    xy32 = p3.x * p3.x - p2.x * p2.x + p3.y * p3.y - p2.y * p2.y
    y0 = (x32 * xy21 - x21 * xy32) / (2 * (y21 * x32 - y32 * x21))
    x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21)
    R = ((p1.x - x0) ** 2 + (p1.y - y0) ** 2) ** 0.5
    return x0, y0, R
    
    
def dis(p1, p2):
    vector1 = np.array([p1.x, p1.y])  
    vector2 = np.array([p2.x, p2.y]) 
    distance = np.linalg.norm(vector1-vector2)
    return  distance

def medLine(x1,y1,x2,y2):
    A = 2*(x2-x1)
    B = 2*(y2-y1)
    C = x1**2-x2**2+y1**2-y2**2
    return A,B,C
def GetIntersectPointofLines(A1,B1,C1,A2,B2,C2):
    m=A1*B2-A2*B1
    if m==0:
        print("Parallel!")
    else:
        x=(C2*B1-C1*B2)/m
        y=(C1*A2-C2*A1)/m
    return x,y
def XORgate(w_a_0,w_a_1,w_b_0,w_b_1): 
    x_00, y_00 = prf(w_a_0, w_b_0)
    x_01, y_01 = prf(w_a_0, w_b_1)
    x_10, y_10 = prf(w_a_1, w_b_0)
    x_11, y_11 = prf(w_a_1, w_b_1)
    p1 = Point(x_00, y_00) 
    p2 = Point(x_01, y_01)
    p3 = Point(x_10, y_10)
    p4 = Point(x_11, y_11)
    A0,B0,C0 = medLine(p1.x,p1.y,p4.x,p4.y)
    A1,B1,C1 = medLine(p2.x,p2.y,p3.x,p3.y)
    x_0, y_0 = GetIntersectPointofLines(A0,B0,C0,A1,B1,C1)
    c0 = Point(x_0, y_0)
    
    dis_00 = dis(p1, c0)
    dis_01 = dis(p2, c0)
    dis_10 = dis(p3, c0)
    dis_11 = dis(p4, c0)
    
    dis_00 = round(dis_00)
    dis_01 = round(dis_01)
    dis_10 = round(dis_10)
    dis_11 = round(dis_11)     
    if not(dis_00==dis_11 and dis_01==dis_10):
        print ("Inequality Error!")
        return None
    w_i_0 = dis_00
    div0 = w_i_0 // 2**63
    if div0>0:
        w_i_0 = w_i_0 / (div0+1)
    w_i_1 = dis_01
    div1 = w_i_1 // 2**63
    if div1>0:
        w_i_1 = w_i_1 / (div1+1)

    w_i_0 = pack('q',w_i_0)
    w_i_1 = pack('q',w_i_1)
    if (w_i_0 == w_i_1):
        print ("Concyclic!")
        return None
    return  w_i_0, w_i_1,x_0, y_0
    
def ANDgate(w_a_0,w_a_1,w_b_0,w_b_1): 

    x_00, y_00 = prf(w_a_0, w_b_0)
    x_01, y_01 = prf(w_a_0, w_b_1)
    x_10, y_10 = prf(w_a_1, w_b_0)
    x_11, y_11 = prf(w_a_1, w_b_1)

    p1 = Point(x_00, y_00) 
    p2 = Point(x_01, y_01)
    p3 = Point(x_10, y_10)
    p4 = Point(x_11, y_11)
    
    x_0, y_0, R = getCircle(p1, p2, p3)
    
    c0 = Point(x_0, y_0)
    R = round(R)
    
    dis_00 = dis(p1, c0)
    dis_01 = dis(p2, c0)
    dis_10 = dis(p3, c0)
    dis_11 = dis(p4, c0)
    
    dis_00 = round(dis_00)
    dis_01 = round(dis_01)
    dis_10 = round(dis_10)
    dis_11 = round(dis_11)    
  
    if not(dis_00==R and dis_01==R and dis_10==R):
        print ("Non-concyclic Error!")
        return None
    
    w_i_0 = R
    div0 = w_i_0 // 2**63
    if div0>0:
        w_i_0 = w_i_0 / (div0+1)
    w_i_1 = dis_11
    div1 = w_i_1 // 2**63
    if div1>0:
        w_i_1 = w_i_1 / (div1+1)

    w_i_0 = pack('q',w_i_0)
    w_i_1 = pack('q',w_i_1)

    if (w_i_0 == w_i_1):
        print ("Concyclic!")
        return None  
    return  w_i_0, w_i_1,x_0, y_0

def ORgate(w_a_0,w_a_1,w_b_0,w_b_1): 
    x_00, y_00 = prf(w_a_0, w_b_0)
    x_01, y_01 = prf(w_a_0, w_b_1)
    x_10, y_10 = prf(w_a_1, w_b_0)
    x_11, y_11 = prf(w_a_1, w_b_1)

    p1 = Point(x_00, y_00) 
    p2 = Point(x_01, y_01)
    p3 = Point(x_10, y_10)
    p4 = Point(x_11, y_11)
    x_0, y_0, R = getCircle(p2, p3, p4)
    c0 = Point(x_0, y_0)
     
    R = round(R)
    
    dis_00 = dis(p1, c0)
    dis_01 = dis(p2, c0)
    dis_10 = dis(p3, c0)
    dis_11 = dis(p4, c0)
    
    dis_00 = round(dis_00)
    dis_01 = round(dis_01)
    dis_10 = round(dis_10)
    dis_11 = round(dis_11) 
    if not(dis_11==R and dis_01==R and dis_10==R):
        print ("Non-concyclic Error!")
        return None
    w_i_0 = dis_00
    div0 = w_i_0 // 2**63
    if div0>0:
        w_i_0 = w_i_0 / (div0+1)
    w_i_1 = dis_01
    div1 = w_i_1 // 2**63
    if div1>0:
        w_i_1 = w_i_1 / (div1+1)

    w_i_0 = pack('q',w_i_0)
    w_i_1 = pack('q',w_i_1)
    
    if (w_i_0 == w_i_1):
        print ("Concyclic!")
        return None

    
    return  w_i_0, w_i_1,x_0, y_0

def NORgate(w_a_0,w_a_1,w_b_0,w_b_1): 
    x_00, y_00 = prf(w_a_0, w_b_0)
    x_01, y_01 = prf(w_a_0, w_b_1)
    x_10, y_10 = prf(w_a_1, w_b_0)
    x_11, y_11 = prf(w_a_1, w_b_1)

    p1 = Point(x_00, y_00) 
    p2 = Point(x_01, y_01)
    p3 = Point(x_10, y_10)
    p4 = Point(x_11, y_11)
    x_0, y_0, R = getCircle(p2, p3, p4)
    c0 = Point(x_0, y_0)
    R = round(R)
    
    dis_00 = dis(p1, c0)
    dis_01 = dis(p2, c0)
    dis_10 = dis(p3, c0)
    dis_11 = dis(p4, c0)
    
    dis_00 = round(dis_00)
    dis_01 = round(dis_01)
    dis_10 = round(dis_10)
    dis_11 = round(dis_11) 
    
    if not(dis_11==R and dis_01==R and dis_10==R):
        print ("Non-concyclic Error!")
        return None
                
    w_i_0 = dis_11
    div0 = w_i_0 // 2**63
    if div0>0:
        w_i_0 = w_i_0 / (div0+1)
    w_i_1 = dis_01
    div1 = w_i_1 // 2**63
    if div1>0:
        w_i_1 = w_i_1 / (div1+1)

    w_i_0 = pack('q',w_i_0)
    w_i_1 = pack('q',w_i_1)
    if (w_i_0 == w_i_1):
        print ("Concyclic!")
        return None

    
    return  w_i_0, w_i_1,x_0, y_0

def NANDgate(w_a_0,w_a_1,w_b_0,w_b_1): 

    x_00, y_00 = prf(w_a_0, w_b_0)
    x_01, y_01 = prf(w_a_0, w_b_1)
    x_10, y_10 = prf(w_a_1, w_b_0)
    x_11, y_11 = prf(w_a_1, w_b_1)
    p1 = Point(x_00, y_00) 
    p2 = Point(x_01, y_01)
    p3 = Point(x_10, y_10)
    p4 = Point(x_11, y_11)

    x_0, y_0, R = getCircle(p1, p2, p3)
    c0 = Point(x_0, y_0)
    R = round(R)
    
    dis_00 = dis(p1, c0)
    dis_01 = dis(p2, c0)
    dis_10 = dis(p3, c0)
    dis_11 = dis(p4, c0)
    
    dis_00 = round(dis_00)
    dis_01 = round(dis_01)
    dis_10 = round(dis_10)
    dis_11 = round(dis_11)  
    
    if not(dis_00==R and dis_01==R and dis_10==R):
        print ("Non-concyclic Error!")
        return None   
          
    w_i_0 = dis_11
    div0 = w_i_0 // 2**63
    if div0>0:
        w_i_0 = w_i_0 / (div0+1)
    w_i_1 = dis_01
    div1 = w_i_1 // 2**63
    if div1>0:
        w_i_1 = w_i_1 / (div1+1)

    w_i_0 = pack('q',w_i_0)
    w_i_1 = pack('q',w_i_1)
    if (w_i_0 == w_i_1):
        print ("Concyclic!")
        return None

    return  w_i_0, w_i_1,x_0, y_0

def XNORgate(w_a_0,w_a_1,w_b_0,w_b_1): 
    x_00, y_00 = prf(w_a_0, w_b_0)
    x_01, y_01 = prf(w_a_0, w_b_1)
    x_10, y_10 = prf(w_a_1, w_b_0)
    x_11, y_11 = prf(w_a_1, w_b_1)

    p1 = Point(x_00, y_00) 
    p2 = Point(x_01, y_01)
    p3 = Point(x_10, y_10)
    p4 = Point(x_11, y_11)

    A0,B0,C0 = medLine(p1.x,p1.y,p4.x,p4.y)
    A1,B1,C1 = medLine(p2.x,p2.y,p3.x,p3.y)
    x_0, y_0 = GetIntersectPointofLines(A0,B0,C0,A1,B1,C1)
    c0 = Point(x_0, y_0)
    
    dis_00 = dis(p1, c0)
    dis_01 = dis(p2, c0)
    dis_10 = dis(p3, c0)
    dis_11 = dis(p4, c0)
    
    dis_00 = round(dis_00)
    dis_01 = round(dis_01)
    dis_10 = round(dis_10)
    dis_11 = round(dis_11) 
    if not(dis_00==dis_11 and dis_01==dis_10):
        print ("Inequality Error!")
        return None    

    w_i_0 = dis(p2, c0)
    div0 = w_i_0 // 2**63
    if div0>0:
        w_i_0 = w_i_0 / (div0+1)
    w_i_1 = dis(p2, c0)
    div1 = w_i_1 // 2**63
    if div1>0:
        w_i_1 = w_i_1 / (div1+1)
    w_i_0 = round(w_i_0)
    w_i_1 = round(w_i_1)
    w_i_0 = pack('q',w_i_0)
    w_i_1 = pack('q',w_i_1)
    if (w_i_0 == w_i_1):
        print ("Concyclic!")
        return None

    return  w_i_0, w_i_1,x_0, y_0
