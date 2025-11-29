## Распределенная система видео аналитики

## api
- **POST /scenario/** - инициализация стейт-машины
- **POST /scenario/<scenario_id>/** - изменение статуса стейт-машины
- **GET /scenario/<scenario_id>/** - информация о текущем статусе сценария
- **GET /prediction/<scenario_id>/** - результаты предсказаний

## orchestrator
- **чтение события (команды)** - получение запроса от api
- **контроль состояния** - сохранение \ изменение \ передача в api (transactional outbox)
- **выполнение действия** - управление runner (сущностями сценариев внутри него)

Поддержка следующих статусов:
- **init_startup** - инициализация запуска
- **in_startup_processing** - промежуточное состояние, олицетворяющее процесс запуска
- **active** - активное состояние \ работа сценария
- **init_shutdown** - инициализация остановки
- **in_shutdown_processing** - промежуточное состояние, олицетворяющее процесс остановки
- **inactive** - выключенное состояние

Жизненный цикл контролируется посредством конечного автомата со следующими переходами:
- init_startup → in_startup_processing → active
- init_shutdown → in_shutdown_processing → inactive

Поддержка:
- **отказоустойчивости** - перезапуск сценария в случае, если тот прекратил свою работу (отсутствие "сердцебиения")
- **масштабирования** - множество runner без дубликатов заданий (сценарий запускается однократно без дополнительных экземпляров только в своем runner)

## runner
- **чтение кадра** - живой поток (rtsp \ onvif \ ...) и\или заготовленное локальное видео
- **препроцессинг (optional)** - подготовка полученного кадра к отправке (BGR2RGB \ resize \ ...)
- **отправка кадра** - отправка кадра в inference
- **получение результата** - чтение результатов с предсказаниями
- **публикация результата** - доступность событий (предсказаний) на стороне api

## inference
- **чтение кадра** - получение кадра
- **предсказание** - inference при помощи модели (mock при отсутствии возможности запуска модели)
- **отправка результатов** - возврат результатов в runner


### Структура проекта:
<pre>
MisisVideoAnalytics/
├── README.md
├── build
│   ├── docker
│   │   ├── api_service
│   │   │   ├── Dockerfile
│   │   │   └── test.Dockerfile
│   │   ├── docker_compose.yml
│   │   ├── images_configuration.json
│   │   ├── inference_service
│   │   │   ├── Dockerfile
│   │   │   └── test.Dockerfile
│   │   ├── orchestrator_service
│   │   │   ├── Dockerfile
│   │   │   └── test.Dockerfile
│   │   └── runner_service
│   │       ├── Dockerfile
│   │       └── test.Dockerfile
│   ├── misis_bootstrap
│   │   ├── pyproject.toml
│   │   ├── setup.py
│   │   ├── source
│   │   │   └── misis_bootstrap
│   │   │       ├── __init__.py
│   │   │       ├── bootstrap.py
│   │   │       ├── main.py
│   │   │       └── package_manager.py
│   │   └── versions.json
│   └── misis_builder
│       ├── pyproject.toml
│       ├── setup.py
│       └── source
│           └── misis_builder
│               ├── __init__.py
│               ├── build_runner.py
│               └── main.py
├── run.sh
└── source
    ├── api_service
    │   └── misis_scenario_api
    │       ├── pyproject.toml
    │       ├── setup.py
    │       ├── source
    │       │   └── misis_scenario_api
    │       │       ├── __init__.py
    │       │       ├── app
    │       │       │   ├── __init__.py
    │       │       │   ├── config.py
    │       │       │   └── web
    │       │       │       ├── __init__.py
    │       │       │       ├── app.py
    │       │       │       ├── routers.py
    │       │       │       └── scenario_service.py
    │       │       ├── database
    │       │       │   ├── __init__.py
    │       │       │   ├── base.py
    │       │       │   ├── database.py
    │       │       │   └── tables
    │       │       │       ├── __init__.py
    │       │       │       ├── outbox.py
    │       │       │       └── scenario.py
    │       │       ├── kafka
    │       │       │   ├── __init__.py
    │       │       │   └── producer.py
    │       │       ├── models
    │       │       │   ├── __init__.py
    │       │       │   ├── bounding_box.py
    │       │       │   ├── constants
    │       │       │   │   ├── __init__.py
    │       │       │   │   ├── command_type.py
    │       │       │   │   ├── kafka_topic.py
    │       │       │   │   └── scenario_status.py
    │       │       │   ├── prediction_response.py
    │       │       │   ├── scenario_create_request.py
    │       │       │   └── scenario_status_response.py
    │       │       ├── outbox
    │       │       │   ├── __init__.py
    │       │       │   └── outbox_worker.py
    │       │       └── s3
    │       │           ├── __init__.py
    │       │           └── s3_client.py
    │       └── tests
    │           ├── __pycache__
    │           ├── conftest.py
    │           └── unit
    │               └── base.py
    ├── inference_service
    │   └── misis_inference
    │       ├── pyproject.toml
    │       ├── setup.py
    │       └── source
    │           └── misis_inference
    │               ├── __init__.py
    │               ├── app
    │               │   ├── __init__.py
    │               │   └── web
    │               │       ├── __init__.py
    │               │       ├── app.py
    │               │       ├── prediction_service.py
    │               │       └── routers.py
    │               └── models
    │                   ├── __init__.py
    │                   ├── bounding_box.py
    │                   └── prediction_response.py
    ├── orchestrator_service
    │   └── misis_orchestrator
    │       ├── pyproject.toml
    │       ├── setup.py
    │       └── source
    │           └── misis_orchestrator
    │               ├── __init__.py
    │               ├── app
    │               │   ├── __init__.py
    │               │   └── config.py
    │               ├── database
    │               │   ├── __init__.py
    │               │   ├── base.py
    │               │   ├── database.py
    │               │   └── tables
    │               │       ├── __init__.py
    │               │       ├── heartbeat.py
    │               │       ├── outbox.py
    │               │       └── scenario.py
    │               ├── health
    │               │   ├── __init__.py
    │               │   └── watchdog.py
    │               ├── kafka
    │               │   ├── __init__.py
    │               │   ├── consumer.py
    │               │   └── producer.py
    │               ├── main.py
    │               ├── models
    │               │   ├── __init__.py
    │               │   ├── constants
    │               │   │   ├── __init__.py
    │               │   │   ├── command_type.py
    │               │   │   ├── kafka_topic.py
    │               │   │   └── scenario_status.py
    │               │   ├── heartbeat.py
    │               │   └── scenario_command.py
    │               ├── orchestrator_service.py
    │               ├── outbox
    │               │   ├── __init__.py
    │               │   └── outbox_worker.py
    │               └── scenario_db_manager.py
    └── runner_service
        └── misis_runner
            ├── pyproject.toml
            ├── setup.py
            └── source
                └── misis_runner
                    ├── __init__.py
                    ├── app
                    │   ├── __init__.py
                    │   ├── config.py
                    │   ├── heartbeat_sender.py
                    │   └── video_processor.py
                    ├── inference_client.py
                    ├── kafka
                    │   ├── __init__.py
                    │   ├── consumer.py
                    │   └── producer.py
                    ├── main.py
                    ├── models
                    │   ├── __init__.py
                    │   ├── bounding_box.py
                    │   └── constants
                    │       ├── __init__.py
                    │       ├── kafka_topic.py
                    │       └── scenario_status.py
                    ├── runner.py
                    └── s3
                        ├── __init__.py
                        └── s3_client.py
</pre>


- Перед запуском добавить в окружение или выполнить:
export REPO_ROOT=/path_to/MisisVideoAnalytics

- Запуск тестов:
./run.sh --test

- локальное развёртывание:
./run.sh