from hugging_face_prototype.image_classifier import initializeImageClassifier
from hugging_face_prototype.utils.helpers import verify_input

def main():
  print("Initializing Hugging Face Prototype")
  initImgClassifier = verify_input("Initialize Image Classifier? (y/n): ", ["y", "n"])
  
  if initImgClassifier == "y":
    initializeImageClassifier()

if __name__ == "__main__":
  main()
