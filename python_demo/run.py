import ChatGLM2

chip_number = 16
engine = ChatGLM2.ChatGLM2()
engine.init(chip_number)
engine.answer("你好")
engine.deinit()