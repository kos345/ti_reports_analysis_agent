from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import DoclingDocument
from docling.datamodel.base_models import InputFormat
from langchain.messages import HumanMessage, SystemMessage

from pydantic import AnyUrl
import pickle
import os

import tempfile
from rich import print
from dotenv import load_dotenv

from src.gigachat_api import analyze_image_langchain

# from src.llm import llm


load_dotenv()
key = os.environ.get('GIGACHAT_API_KEY')
openai_key = os.environ.get('OPENAI')

def import_pickle(file:str) -> DoclingDocument:
    """
    Import DoclingDocument from pickle file
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def export_pickle(data: DoclingDocument, filename:str)-> None:
    """
    Export data to pickle file
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def export_md(text:str, path:str) -> None:
    """
    Export text to markdown file
    """
    with open(path, 'w') as f:
        f.write(text)

def load_from_exist_file(file:str)-> DoclingDocument | str | None:
    """
    Load from existing file
    """
    
    # print(f"File {file} exists")
    if Path(file).as_posix().endswith('pkl'):
        return import_pickle(file)
    if Path(file).as_posix().endswith('.md'):
        with open( file, 'r' ) as f:
            return f.read()
 

def parse_pdf_docling(state):
    pdf_path = state["pdf_path"]
    name = Path(pdf_path).stem
    out_dir = Path("work")
    out_dir.mkdir(parents=True, exist_ok=True)
    # print(f"{Path(pdf_path).exists() = }")
    
    '''Этот блок не понятен'''
    # если уже есть файл, просто грузим из файла
    # if Path(pdf_path).exists():
    #     # print("file exist")
    #     return {
    #         "docling_doc": load_from_exist_file(out_dir.joinpath(name + ".pkl").as_posix()), 
    #         "doc_markdown": load_from_exist_file(pdf_path)
    #         }

    # если уже есть готовые pkl и md
    pkl_path = out_dir / f"{name}.pkl"
    md_path = out_dir / f"{name}.md"
    if pkl_path.exists() and md_path.exists():
        return {
            "docling_doc": load_from_exist_file(pkl_path.as_posix()),
            "doc_markdown": load_from_exist_file(md_path.as_posix()),
        }


    converter = DocumentConverter()
    # если нет файла — парсим и сохраняем
    res = converter.convert(str(pdf_path))
    doc = res.document
    doc_md = doc.export_to_markdown(page_break_placeholder="<!-- PAGE_BREAK -->")  
    name = Path(pdf_path).stem
    out_dir = Path("work")

    # export_pickle(doc, out_dir.joinpath(name + ".pkl").as_posix())
    # export_md(doc_md, out_dir.joinpath(name + ".md").as_posix())
    # return {"docling_doc": doc,"doc_markdown": doc_md}

    export_pickle(doc, pkl_path.as_posix())
    export_md(doc_md, md_path.as_posix())
    return {"docling_doc": doc, "doc_markdown": doc_md}


def parse_pdf_docling_with_images(state):
    pdf_path = state["pdf_path"]
    name = Path(pdf_path).stem
    img_path = name + "_img.md"

    out_dir = Path("work")
    out_dir.mkdir(parents=True, exist_ok=True)
    # если уже есть файл с изображениями, просто грузим из файла
    # print(out_dir.joinpath(Path(img_path)).exists())
    if out_dir.joinpath(Path(img_path)).exists():
        # print("file exist")
        return {"docling_doc": load_from_exist_file(out_dir.joinpath(name + ".pkl").as_posix()), 
                "doc_markdown": load_from_exist_file(out_dir.joinpath(name + "_img.md").as_posix())}

    # OCR/пайплайн — настраиваем через PipelineOptions [web:61]
    pipeline_options = PdfPipelineOptions(enable_remote_services=True)
    pipeline_options.do_picture_description = True


    pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        url=AnyUrl("https://api.openai.com/v1/chat/completions"),
        headers={
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        },
        params={
            "model": "gpt-4o-mini",
            "max_completion_tokens": 200,
            "temperature": 0.0,
            "seed": 42,
        },
        prompt="Подробно опиши данные на изображении. Если есть текст/числа — извлеки их максимально дословно.",
        timeout=90,
        )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    # Варианты зависят от окружения/установленных OCR движков; ключевое — что настройки есть [web:61]

    
    res = converter.convert(str(pdf_path))
    doc = res.document

    doc_md = doc.export_to_markdown()  # [web:58]
    # print(f"[white]{doc_md[:400]}[/white]")

    
    export_pickle(doc, out_dir.joinpath(name + ".pkl").as_posix())
    export_md(doc_md, out_dir.joinpath(name + "_img.md").as_posix())
    return {"docling_doc": doc,"doc_markdown": doc_md }

DEFAULT_IMAGE_PROMPT = '''
Подробно опиши данные на изображении. 
Если есть текст, числа, графики, диаграммы или схемы — извлеки их максимально дословно.
Отвечай на русском языке.
'''

def make_parse_pdf_gigachat_postprocess(prompt: str = DEFAULT_IMAGE_PROMPT):
    """
    Фабричная функция. Возвращает узел LangGraph для парсинга PDF
    с постобработкой изображений через GigaChat (без прокси-сервера).

    Принцип работы:
      1. docling парсит PDF и сохраняет позицию каждой картинки как <!-- image --> в markdown.
      2. doc.pictures и <!-- image --> в markdown идут в одном порядке (1:1).
      3. Для каждого PictureItem вызывается analyze_image_langchain().
      4. Первое вхождение <!-- image --> заменяется описанием (count=1),
         затем берётся следующая картинка — так описания встают строго на свои места.

    Args:
        prompt: промпт для анализа изображений через GigaChat

    Returns:
        Функция-узел LangGraph с сигнатурой (state: dict) -> dict
    """
    def _node(state: dict) -> dict:
        pdf_path = state["pdf_path"]
        name = Path(pdf_path).stem
        out_dir = Path("work")
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / f"{name}_gigachat_v2.md"
        pkl_path = out_dir / f"{name}_gigachat_v2.pkl"

        # Кэш: если файлы уже есть — не перепарсиваем
        if md_path.exists() and pkl_path.exists():
            return {
                "docling_doc": load_from_exist_file(pkl_path.as_posix()),
                "doc_markdown": load_from_exist_file(md_path.as_posix()),
            }

        # Парсим PDF с извлечением PIL-изображений (без вызова внешнего LLM)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        res = converter.convert(str(pdf_path))
        doc = res.document

        # Базовый markdown: каждое изображение → <!-- image --> на своём месте в потоке текста
        doc_md = doc.export_to_markdown(page_break_placeholder="<!-- PAGE_BREAK -->")

        # Постобработка: описываем каждую картинку и вставляем на её место
        # doc.pictures и вхождения <!-- image --> в markdown совпадают по порядку
        for picture in doc.pictures:
            pil_image = picture.get_image(doc)
            if pil_image is None:
                # нет данных изображения — оставляем плейсхолдер нетронутым
                continue

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_image.save(tmp.name)
                tmp_path = tmp.name

            try:
                description = analyze_image_langchain(tmp_path, prompt)
                replacement = f"\n**[Изображение]**: {description}\n"
            except Exception as e:
                replacement = f"<!-- image: ошибка описания — {e} -->"
            finally:
                os.unlink(tmp_path)

            # count=1: заменяем строго первое вхождение — это и есть текущая картинка в тексте
            doc_md = doc_md.replace("<!-- image -->", replacement, 1)

        export_pickle(doc, pkl_path.as_posix())
        export_md(doc_md, md_path.as_posix())
        return {"docling_doc": doc, "doc_markdown": doc_md}

    return _node