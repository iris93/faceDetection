#coding:utf-8
import cv2#因为还是需要用到opencv里面的一些函数
from faceModel import Model#这个文件为上一篇的文件代码，需要引入里面的一个Model对人脸进行预测

CAMERA_ID=0#默认摄像头id为0

if __name__ == '__main__':
    model=Model()
    model.load_model(file_path='./face.model.h5')#加载分类器

    color = (0,255,0)#框的颜色
    cap=cv2.VideoCapture(CAMERA_ID)#打开摄像头，读取视频流

    cassade_path='haarcascade_frontalface_default.xml'#opencv人脸分类器路径，分类器路径可以修改

    while True:
        ok,frame=cap.read()

        frame_grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#灰度化

        cassade=cv2.CascadeClassifier(cassade_path)#加载分类器

        faceRects=cassade.detectMultiScale(frame_grey,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))#利用opencv获取视频frame里面的所有人脸
        if len(faceRects)>0:
            for faceRect in faceRects:#遍历每个人脸
                x,y,w,h=faceRect#获取人脸的框的大小
                image=frame[y:y+h,x:x+w]#从视频一帧中截取只有人脸的一个片段
                faceId=model.face_predict(image)#调用函数进行预测
                print faceId
                if faceId==1:#id=0表示为本人，其他不是本人
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)#画框

                    cv2.putText(frame,'your grace',
                                (x+30,y+30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255,0,255),
                                2)#文字框，表示已经识别了本人
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    cv2.putText(frame, '',
                                (x + 30, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 255),
                                2)

                    pass#否则跳过

        cv2.imshow('识别朕',frame)
        k=cv2.waitKey(10)
        if k& 0xFF==ord('q'):
            break
    cap.release()#释放资源
    cv2.destroyAllWindows()