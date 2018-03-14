
# Dedicacion: alta, media, baja.
# Dificultad: alta, media, baja. 
# Horario: matutino, nocturno. 
# Humedad: alta, media, baja. 
# HumorDoc: bueno, malo.

from enum import Enum

NO_VALUE =	"0"
ANY_VALUE = "?"

class Dedicacion(Enum):
	ALTA = 3
	MEDIA = 2
	BAJA = 1

class Dificultad(Enum):
	ALTA = 3
	MEDIA = 2
	BAJA = 1

class Horario(Enum):
	MATUTINO = 2
	NOCTURNO = 1

class Humedad(Enum):
	ALTA = 3
	MEDIA = 2
	BAJA = 1	
	

class HumorDoc(Enum):
	BUENO = 2
	MALO = 1


