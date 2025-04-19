import matplotlib.pyplot as plt

def imshow(img):
    plt.imshow(img)
    plt.show()
    
def bi_imshow(image):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()