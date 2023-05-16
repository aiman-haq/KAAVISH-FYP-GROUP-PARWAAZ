# KAAVISH-PARWAAZ

## Abtract 

The aim of our project is to design, develop and implement
advanced computer vision techniques (algorithms) to extract
out meaningful information from drone-imagery in real-time.
Our project is divided into three modules, as follows:
● Face recognition
● Object detection (i.e. trees and vehicles)
● Object counting
For this project, we have collaborated with a company
Woot-Tech. We have developed algorithms for them, which
operate both as manufacture, and operations, of bespoke drone
technology products for industrial-strength commercial use.

## Module 1: Face Recognition

Face module is divided into following sub-modules:
Module-1 deals with reading the data from a custom
dataset, which includes gathering facial feature encodings of
all the images present in the data-set. Module-2 deals with
reading the live video input, detect the faces present in it and
gather the data of facial feature encodings for faces present
in live video frame. Module-3 deals with the comparison of
Module-1 and Module-2 data, and further evaluate the facial
encodings and generate the processed output video frame
with recognized faces on it.

## Module 2: Object Recognition

To implement our module, after our brief literature research, we
used deep learning YOLO model to perform Object Recognition,
which is faster than most techniques. We used a pre-trained
dataset on YOLOV3 for vehicles and a customized trained
dataset for trees, further input is sent to deep neural network
once, which outputs the input image with bounding boxes and
confidence that encloses the detected object along with their
class labels. In this process, low confidence objects are
being excluded to have better accuracy.

## Note

- The Main Branch doesnt have any files, kindly, refer to our two other branches to have access to both Modules.
- Due to exceed size of the files, Yolo-Coco folder that carries weighted and configuration files of the trained dataset
can be found at given OneDrive link:
https://habibuniversity-my.sharepoint.com/:f:/g/personal/ah04318_st_habib_edu_pk/ErwhaTyjT6dPqkkQwhHScjEBO9cX2HJGt35_jVnazZk15Q?e=bm7Nme
