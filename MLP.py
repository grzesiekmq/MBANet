import dl_numpy as DL
import utilities
import numpy as np

if __name__ == "__main__":
    batch_size = 2
    num_epochs = 10

    num_classes = 3
    hidden_units = 6

    xTxt = ["beer", "milk", "bread"]
    yTxt = ["chips", "cereals", "butter"]

    def label_encoder(list):
        l = np.arange(len(list))
        return l

    x_encoded = label_encoder(xTxt)
    x = x_encoded.reshape(1, -1)

    y = label_encoder(yTxt)
    # print(x)

    # MLP
    model = utilities.Model()
    model.add(DL.Linear(num_classes, hidden_units))
    model.add(DL.ReLU())
    model.add(DL.Linear(hidden_units, num_classes))
    optim = DL.SGD(model.parameters, lr=1.0, weight_decay=0.001, momentum=0.9)
    loss_fn = DL.SoftmaxWithLoss()
    model.fit(x, y, batch_size, num_epochs, optim, loss_fn)

    lookup = dict(zip(xTxt, x_encoded))

    print("probs", DL.probs[-1][0])

    def pred(text):
        if lookup[text] == x_encoded[lookup[text]]:
            pred_val = DL.probs[-1][0][lookup[text]]
            return list(DL.probs[-1][0]).index(pred_val)

    def predictions(texts):
        pred_vals = []
        for text in texts:
            if lookup[text] == x_encoded[lookup[text]]:
                pred_val = DL.probs[-1][0][lookup[text]]
                pred_vals.append(list(DL.probs[-1][0]).index(pred_val))
        return pred_vals

    preds = predictions(xTxt)
    print('preds', preds)

    text = input("type name of product (e.g beer): ")
    test = pred(text)



    print("predicted label index e.g 0 - chips: ", test)

    predicted = ""
    if test == y[test]:
        predicted = yTxt[test]

    print("do you want to add", predicted + "?")

    accuracy = np.sum(y == preds) / len(y)

    print("Model Accuracy = {}".format(accuracy))
