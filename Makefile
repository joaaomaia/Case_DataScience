all: setup data eda model report

setup:
	@echo "Configurar ambiente, instalar dependências"

data:
	@echo "Processar dados"

eda:
	@echo "Executar análise exploratória"

model:
	@echo "Treinar e avaliar modelo"

report:
	@echo "Gerar relatório final"
