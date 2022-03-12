import cv2
import os
import random
import xml.etree.ElementTree as ET

def generte_boundingbox_from_mask(mask):
    """
    Generate bounding box from mask
    """
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find bounding box
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h


def generte_boundingbox_from_mask_list(mask,
                                  mask_bb_path,
                                  show=False):
    """
    Generate bounding box from mask
    """
    # get contours
    result = mask.copy()
    bb_box_list =[]
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find bounding box
    x, y, w, h = cv2.boundingRect(contours[0])
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        bb_box_list.append[(x,y,w,h)]
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("x,y,w,h:",x,y,w,h)
    if show:
        show_image(result)
        # save resulting image
        cv2.imwrite(mask_bb_path, result)
     
    
    return bb_box_list

def read_image(image_path):
    """
    Read image
    """
    image = cv2.imread(image_path)
    return image

def plot_bounding_box_on_image(image, x, y, w, h):
    """
    Plot bounding box on image
    """
    # convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

def read_mask(mask_path):
    """
    Read mask
    """
    mask = cv2.imread(mask_path, 0)
    return mask

def show_image(image):
    """
    Show image
    """
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def write_image(image, image_path):
    """
    Write image
    """
    cv2.imwrite(image_path, image)
    
def create_list_tuple(image_path, mask_path, mask_bb_path):
    """
    Create list of tuples
    """
    return [(image_path, mask_path, mask_bb_path)]

def read_all_files_from_directory(directory):
    """
    Read all files from directory
    """
    return os.listdir(directory)

def read_all_images_from_directory(directory):
    """
    Read all images from directory
    """
    return [f for f in os.listdir(directory) if f.endswith('.jpg')]

def view_all_images_from_directory(directory):
    """
    View all images from directory
    """
    for f in os.listdir(directory):
        if f.endswith('.jpg'):
            image = read_image(os.path.join(directory, f))
            show_image(image)

def show_random_images_from_directory(directory, n):
    """
    Show random images from directory
    """
    for i in range(n):
        f = random.choice(os.listdir(directory))
        image = read_image(os.path.join(directory, f))
        show_image(image)
        



def write_pascal_xml_format(image, mask, mask_bb, image_path, mask_path, mask_bb_path):
    """
    Write pascal xml format
    """
    # Create an empty element tree
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'VOC2007'
    ET.SubElement(annotation, 'filename').text = os.path.basename(image_path)
    ET.SubElement(annotation, 'path').text = image_path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'The VOC2007 Database'
    ET.SubElement(source, 'annotation').text = 'PASCAL VOC2007'
    ET.SubElement(source, 'image').text = 'flickr'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(image.shape[1])
    ET.SubElement(size, 'height').text = str(image.shape[0])
    ET.SubElement(size, 'depth').text = str(image.shape[2])
    ET.SubElement(annotation, 'segmented').text = '0'
    # Create object element
    object = ET.SubElement(annotation, 'object')
    ET.SubElement(object, 'name').text = 'person'
    ET.SubElement(object, 'pose').text = 'Unspecified'
    ET.SubElement(object, 'truncated').text = '0'
    ET.SubElement(object, 'difficult').text = '0'
    bndbox = ET.SubElement(object, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(mask_bb[0])
    ET.SubElement(bndbox, 'ymin').text = str(mask_bb[1])
    ET.SubElement(bndbox, 'xmax').text = str(mask_bb[2])
    ET.SubElement(bndbox, 'ymax').text = str(mask_bb[3])