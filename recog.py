import numpy as np
import multiprocessing
import time
import cv2
import re
import os
import glob
import psycopg2
from pathlib import Path
import argparse
import configparser

# Функция взаимодействия с базой данных
def dbtalk(host_u: str, port_u: int, user_u: str, password_u: str, database_u: str, wr, command, elements=None):
    # Подключаемся к базе
    db = psycopg2.connect(host=host_u, port=port_u, user=user_u, password=password_u, database=database_u)

    with db:
        cursor = db.cursor()
        query = str(command)
		
        # Примение команды в зависимости от количества аргументов
        if elements!=None:
            cursor.executemany(query, elements)
        else:
            cursor.execute(query)
		
        # Чтение или запись в базу
        # wr = w - write
        # wr = r - read
        if wr=='w':
            return db.commit()
        elif wr=='r':
            return cursor.fetchall()


# Функция обработки кадров
def video_processing(stream_name, proc_id, srt, labelfile, weightsfile, configfile, semaphore, result_id, skip_f, number_of_processes, number_of_batches,host_i, port_i, user_i, password_i, database_i):

    # Декремент значения semaphore
    semaphore.acquire()
    skip_ef=skip_f

    # Пути к конфигурационным файлам
    path_to_label = str(labelfile)
    path_to_weights = str(weightsfile)
    path_to_config = str(configfile)

    # Получение имён классов из базы
    LABELS_ID = dbtalk(host_i, port_i, user_i, password_i, database_i, 'r', 'SELECT id_object_type, sysname from uni.object_types')
    
    # Загружаем имена классов
    labelsPath = os.path.sep.join([path_to_label])
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


    weightsPath = os.path.sep.join([path_to_weights])
    configPath = os.path.sep.join([path_to_config])

    # Подгрузка модели нейронной сети
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Подключаемся к видеозаписи
    vs = cv2.VideoCapture(stream_name)

    # Размеры фрейма
    (W, H) = (None, None)

    # Обрабатывается каждый ef-1 фрейм (прореживание)
    ef = skip_ef  
    try:
        prop = cv2.CAP_PROP_FRAME_COUNT
        # Общее число всех кадров
        total_es = int(vs.get(prop))
        # Число обрабатываемых кадров
        total = total_es // ef
        print("Общее количество кадров {0} потока {1}".format(total, proc_id))
    except:
        print("Невозможно определить число кадров")
        total = -1
        
    # Счетчик кадров (прореживание)
    count = 0  
    elements = []
    while True:
        # Считывание фреймов
        (grabbed, frame) = vs.read()

        # Сброс, если фреймов больше нет
        if not grabbed:
            break

        # Проверка кадра (пропуск/обработка)
        if count % ef == 0:
            # Получение текущего времени фрейма в мс и перевод в сек относительно начального времени в имени видеофайла
            timestamp = vs.get(cv2.CAP_PROP_POS_MSEC)
            srt_sec = int(srt + '000')
            timestamp_r = round(timestamp) + int(srt_sec)

            # Определение размеров фрейма
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # Отсчёт времени для обработки
            start_time_full = time.time()

            # Перевод нейросети на видеокарту
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            # Получение blob фрейма
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (412, 412), swapRB=True, crop=False)
            b = cv2.UMat(blob)
            net.setInput(b)
            
            # Время обработки нейросетью
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # Откидываем детекции с малой уверенностью
                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # Вычислим координату верхнего левого угла
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        cv2.UMat(boxes.append([x, y, int(width), int(height)]))
                        cv2.UMat(confidences.append(float(confidence)))
                        cv2.UMat(classIDs.append(classID))

            # Использование Nonmaxima supress
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])

                    # Нормировка координат детектирующих прямоугольников
                    x_opt = (10000 / W) * x
                    y_opt = (10000 / H) * y

                    x_opt_r = round(x_opt)
                    y_opt_r = round(y_opt)

                    (w, h) = (boxes[i][2], boxes[i][3])

                    # Нормировка размеров детектирующих прямоугольников
                    w_opt = (10000 / W) * w
                    h_opt = (10000 / H) * h

                    w_opt_r = round(w_opt)
                    h_opt_r = round(h_opt)

                    text = LABELS[classIDs[i]]

                    lid_dict = dict(LABELS_ID)
                    dict_rev = dict([(value, key) for key, value in lid_dict.items()])
                    obj_id = dict_rev[text]

                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    cv2.imwrite("/media/user/Data/YOLO_mine/yolo_project/test/{0}_{1}_{2}_{3}_{4}.jpg".format(timestamp_r, x_opt_r, y_opt_r, w_opt_r, h_opt_r), frame)


                   
                    if count!=0 and count % 300 == 0:
                        # Запись данных об объектах в кадре в базу данных
                        dbtalk(host_i, port_i, user_i, password_i, database_i, 'w',
                                           'INSERT INTO detector.objects(id_camera, id_object_type, id_object, object_time, x, y, w, h) VALUES(%s, %s, %s, %s, %s, %s, %s, %s)', elements)
                        elements.clear()

                    else:
                        # Накопление строки данных
                        elements.append((result_id, obj_id, 0, timestamp_r, x_opt_r, y_opt_r, w_opt_r, h_opt_r))

                boxes.clear()
                confidences.clear()
                classIDs.clear()

                total_time = float(time.time() - start_time_full)
                if total > 0:
                    elap = (end - start)
                    print("#############################################")
                    print('size', H, W)
                    print("Процесс номер", proc_id)
                    print("Общее количество кадров {0}".format(total))
                    print("Обработка одного кадра составляет {:.4f} секунд".format(elap))
                    print('Времени на полный кадр', total_time)
                    print("#############################################")


        count += 1

    vs.release()
    print('IM OK')

    # Запись оставшихся данных об объектах в кадре в базу данных
    dbtalk(host_i, port_i, user_i, password_i, database_i, 'w',
           'INSERT INTO detector.objects(id_camera, id_object_type, id_object, object_time, x, y, w, h) VALUES(%s, %s, %s, %s, %s, %s, %s, %s)',
           elements)
    elements.clear()
    
    # Инкремент значения semaphore
    semaphore.release()

# Главная функция конфигурирования и подготовки системы
def testsys(directory, skip_f, number_of_processes, number_of_batches, date, ip_cam, channel, weights_path, weights_i, labels_i, config_i, host_i, port_i, user_i, password_i, database_i):

    # Приведение пути к приемлемому для текущей ОС
    directory = str(Path(directory))
    
    # Определение длины пути
    num_dir = directory.split('/')
    nd = len(num_dir)

    # Максимальное значение параллельных процессов
    max_val = number_of_processes

    # Инициализация semaphore
    semaphore = multiprocessing.Semaphore(value=max_val)

    # Определение пользовательского пути для ip камеры
    if ip_cam!=None:
        if channel!=None:
            arglob = glob.glob(str(Path("{0}/{1}/********/{3}/*.mp4".format(directory, ip_cam, channel))))
        else:
            arglob = glob.glob(str(Path("{0}/{1}/********/**/*.mp4".format(directory, ip_cam))))
    
    # Определение пользовательского пути для даты
    if date!=None:
        if channel!=None:
            arglob = glob.glob(str(Path("{0}/****/{1}/{2}/*.mp4".format(directory, date, channel))))
        else:
            arglob = glob.glob(str(Path("{0}/****/{1}/**/*.mp4".format(directory, date))))
	
    # Определение пользовательского пути для канала
    if channel!=None:
        arglob = glob.glob(str(Path("{0}/****/********/{1}/*.mp4".format(directory,channel))))
    
    # Определение общего пути
    if date==None and channel==None and ip_cam==None:
        arglob = glob.glob(str(Path("{}/****/********/**/*.mp4".format(directory))))
    
    
    allproc = []
    
    # Оределение количества видеозаписей, зранимых в ОЗУ
    batchsize = number_of_batches
    for i in range(0, len(arglob), batchsize):
        allproc.clear()
        print('NEW ROUND', allproc)
        batch = arglob[i:i + batchsize]
        for id, k in enumerate(batch):
            list1 = k.split('/')
            print(list1)

            # Путь к общей папке
            wd_ind=nd+1
            which_dir = '/'.join(list1[:wd_ind])
            #print(which_dir)

            sr_ind=nd+2
            sr = list1[sr_ind].strip()
            #print(sr)

            # Автоматический поиск конфигурационный файлов по стандартной иерархии
            if weights_path=='auto':
			
                if os.path.isdir(str(Path("{0}/Vesa/{1}".format(which_dir, sr)))):
                    labelfile = ''.join(glob.glob(str(Path("{0}/Vesa/{1}/*.names".format(which_dir, sr)))))
                    configfile = ''.join(glob.glob(str(Path("{0}/Vesa/{1}/*.cfg".format(which_dir, sr)))))
                    weightsfile = ''.join(glob.glob(str(Path("{0}/Vesa/{1}/*.weights".format(which_dir, sr)))))

                else:
                    labelfile = ''.join(glob.glob(str(Path("{}/Vesa/*.names".format(which_dir)))))
                    configfile = ''.join(glob.glob(str(Path("{}/Vesa/*.cfg".format(which_dir)))))
                    weightsfile = ''.join(glob.glob(str(Path("{}/Vesa/*.weights".format(which_dir)))))
					
            # Ручной ввод путей к конфигурационным файлам
            elif weights_path=='manual':
                labelfile = labels_i
                configfile = config_i
                weightsfile = weights_i
		
            ndf=nd+3
            search = re.search(r'\_\d+\_', list1[ndf])
            srt = search.group()[1:-1]
            ip_dir = list1[nd].strip()
            #print(ip_dir)
            #print(sr)
            result_idt = dbtalk(host_i, port_i, user_i, password_i, database_i, 'r',
                   "SELECT id_camera FROM uni.cameras WHERE (camera_data::json->>'channelid') = '{0}' and (camera_data::json->>'server') = '{1}' and deleted = false limit 1".format(sr, ip_dir))
            #print(result_idt)
            # получение номера камеры по ip и каналу из базы данных
            #result_id = result_idt[0][0]
            result_id=1

            # Инициализация процессов обработки
            process = multiprocessing.Process(target=video_processing,
                                            args=(k, 
                                                  id, 
                                                  srt, 
                                                  labelfile, 
                                                  weightsfile, 
                                                  configfile, 
                                                  semaphore, 
                                                  result_id, 
                                                  skip_f, 
                                                  number_of_processes, 
                                                  number_of_batches,
                                                  host_i,
                                                  port_i,
                                                  user_i,
                                                  password_i,
                                                  database_i))


            allproc.append(process)
            process.start()
            
        # Завершение процессов
        for process in allproc:
            process.join()
        print(allproc)
    print('ready with set')

def main():
# Команды для работы с системой
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_videoset',    type=str, required=True,  help='path to set of videos')
    parser.add_argument('-i', '--ini_file',            type=str, required=True,  help='path to ini file with DB info')
    parser.add_argument('-f', '--skip_frames',         type=int, default=6,      help='how many frames skip')
    parser.add_argument('-n', '--number_of_processes', type=int, default=7,      help='number of parallel processes')
    parser.add_argument('-b', '--number_of_batches',   type=int, default=45,     help='how many videos save in RAM')
    parser.add_argument('-d', '--date',                type=str, default=None,   help='choose date to process')
    parser.add_argument('-ip', '--ip_cam',             type=str, default=None,   help='choose ip of a camera to process')
    parser.add_argument('-c', '--channel',             type=int, default=None,   help='choose channel to process')
    parser.add_argument('-w', '--weights_path',        type=str, choices=['auto', 'manual'], help='find wheights automaticall or set them manually')
	
    parser.add_argument('-iw', '--ini_wlc',            type=str, default=None,  help='path to ini file with weights, labels and config files')

    args = parser.parse_args()
    

    config = configparser.ConfigParser()
    config.read(args.ini_file)

    host_i     = str(config.get("Settings", "host"))
    port_i     = config.get("Settings", "port")
    user_i     = str(config.get("Settings", "user"))
    password_i = str(config.get("Settings", "password"))
    database_i = str(config.get("Settings", "database"))

    if args.weights_path=='manual':
       config_p = configparser.ConfigParser()
       config_p.read(args.ini_wlc)

       weights_i = config_p.get("Paths", "path_weights")
       labels_i  = config_p.get("Paths", "path_labels")
       config_i  = config_p.get("Paths", "path_config")
    else:
       weights_i = None
       labels_i  = None
       config_i  = None
     



    # Передача параметров в главную функцию
    testsys(args.path_to_videoset,
	        args.skip_frames,
		args.number_of_processes,
			args.number_of_batches,
			args.date,
                        args.ip_cam,
			args.channel,
			args.weights_path,
            weights_i,
            labels_i,
            config_i,
            host_i,
            port_i,
            user_i,
            password_i,
            database_i
			)


if __name__ == '__main__':
    # Время полного цикла обработки
    start_time_full = time.time()
    main()
    total_time = float(time.time() - start_time_full)
    print(total_time)
