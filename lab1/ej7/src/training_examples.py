#Dedicación	Dificultad	Horario		Humedad		Humor Doc	Salva
#1	Alta	Alta		Nocturno	Media		Bueno		SÍ
#2	Baja	Media		Matutino	Alta		Malo		NO
#3	Media	Alta		Nocturno	Media		Malo		SÍ
#4	Media	Alta		Matutino	Alta		Bueno		NO

#<?, Alta, Nocturno, Media, ?>

from random import randint
from constants import Dedicacion, Dificultad, Horario, Humedad, HumorDoc

def example1():
	return [
		((Dedicacion.ALTA,   Dificultad.ALTA,  Horario.NOCTURNO, Humedad.MEDIA, HumorDoc.BUENO), 1),
		((Dedicacion.BAJA,   Dificultad.MEDIA, Horario.MATUTINO, Humedad.ALTA,  HumorDoc.MALO),  0),
		((Dedicacion.MEDIA,  Dificultad.ALTA,  Horario.NOCTURNO, Humedad.MEDIA, HumorDoc.MALO),  1),
		((Dedicacion.MEDIA,  Dificultad.ALTA,  Horario.MATUTINO, Humedad.ALTA,  HumorDoc.BUENO), 0)
	];


def random_training_examples(n, cant_pos):
	
	list = []
	while n > 0 and cant_pos > 0:
		instance = (Dedicacion(randint(1,3)), Dificultad(randint(1,3)), Horario(randint(1,2)), Humedad(randint(1,3)), HumorDoc(randint(1,2)))
		c = target(instance)
		if instance not in list:
			list.append((instance, c))
			if c == 1 :
				cant_pos = cant_pos - 1
			n = n - 1
	return list



#<?, Media, ?, ?, ?>
def target(instance):
	return instance[1] == Dificultad.MEDIA

