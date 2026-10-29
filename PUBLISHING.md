# Выгрузка на GitHub

В текущем рабочем каталоге исходники, README и LICENSE находятся прямо в корне проекта. В корневом Git-репозитории уже настроены `origin`, ветка `main` и ее связь с `origin/main`. Подготовка файлов не отправляет их на GitHub.

После локального коммита для этой рабочей копии достаточно выполнить из корня:

```sh
git push
```

Для отдельной новой копии из архива:

1. Распакуйте архив и проверьте его командами из README.
2. Создайте пустой репозиторий `ASTC-KV` в своем аккаунте. Не добавляйте второй README или LICENSE при создании.
3. Выполните команды ниже из папки `ASTC-KV`, заменив `YOUR_ACCOUNT` на свой аккаунт. Эти команды изменяют Git и отправляют данные только при вашем запуске.

```sh
git init -b main
git add .
git status --short
git commit -m "Archive ASTC-KV research snapshot 2026-10-29"
git remote add origin https://github.com/YOUR_ACCOUNT/ASTC-KV.git
git push -u origin main
```

В `.gitignore` исключены зависимости, веса, окружения, локальные сборки, `.env`, настройки редактора, `github-release/`, резервные копии и оставшийся локальный каталог `astc-kv/` с прежними зависимостями и сборками. Вложенного Git-репозитория в `astc-kv/` больше нет. Его прежняя история сохранена в локальной резервной копии. Численные результаты намеренно включены. Перед новым `git commit` просмотрите состав файлов. Вместо терминала можно использовать GitHub Desktop для корня проекта.

Архив содержит больше 100 файлов, поэтому выгрузка всей папки через Git или GitHub Desktop удобнее веб-формы. По [документации GitHub](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository), веб-форма принимает до 100 файлов за один раз и до 25 MiB на файл. Проверка пакета контролирует размеры файлов; Git LFS для текущего снимка не требуется. ZIP следует распаковать для репозитория с доступными для просмотра исходниками. Сам ZIP можно приложить отдельно к GitHub Release.

Предлагаемое описание репозитория:

```text
ASTC KV-cache research on Apple GPUs: Metal kernels, reproducible measurements and negative results. Research archive, not a production inference accelerator.
```

Предлагаемые topics: `metal`, `apple-silicon`, `astc`, `kv-cache`, `attention`, `research`, `benchmark`.

Предлагаемое имя GitHub Release: `Research snapshot 2026-10-29`. Лицензия оригинального кода: MIT. Статус: исследование завершено на данном снимке; преимущества полного inference и научная новизна не установлены.
