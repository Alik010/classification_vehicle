import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto

onnx_path = "onnx/cls_vehicle.onnx"
# Загрузка существующей модели ONNX
model = onnx.load(onnx_path)

graph = model.graph

# Добавляем Softmax узел
softmax_output_name = "softmax_output"
softmax_node = helper.make_node(
    'Softmax',
    inputs=[graph.output[0].name],  # Вход для softmax – это текущий выход модели
    outputs=[softmax_output_name],
    axis=1  # Softmax по оси классов
)

# Добавляем ArgMax узел
argmax_output_name = "argmax_output"
argmax_node = helper.make_node(
    'ArgMax',
    inputs=[softmax_output_name],  # Вход для ArgMax – это выход softmax
    outputs=[argmax_output_name],
    axis=1,  # ArgMax по оси классов
    keepdims=0  # Возвращаем индексы классов без дополнительного измерения
)

# Обновляем выходы модели (после добавления ArgMax)
graph.node.append(softmax_node)
graph.node.append(argmax_node)

# Изменение типа и названия выходного тензора модели
graph.output[0].type.tensor_type.elem_type = TensorProto.INT64  # Обновляем тип на int64
graph.output[0].name = argmax_output_name  # Устанавливаем новый выход

# Сохранение модифицированной модели
onnx.save(model, "onnx/modified_model_with_softmax_argmax.onnx")

print("Модель успешно модифицирована и сохранена.")