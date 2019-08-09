import data_reader as dr
import matplotlib.pyplot as plt
import pickle


def main():
    with open(dr.data('losses.pickle'), 'rb') as f:
        losses = pickle.load(f)

    plt.plot(losses)
    plt.xlabel("Cantidad de instancias (en miles)")
    plt.ylabel("Función de costo")
    plt.title("Evaluación de la función de costo sobre el dataset de entrenamiento")
    plt.savefig(dr.data('losses.png'))
    plt.show()


if __name__ == '__main__':
    main()
