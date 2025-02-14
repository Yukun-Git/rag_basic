from collections import deque

import camelot
import pandas as pd
import pdfplumber


def extract_table_by_pdfplumber(pdf_path, title_search_height=50):
    """
    改进版表格提取函数
    :param pdf_path: PDF文件路径
    :param title_search_height: 标题搜索范围（像素，默认表格上方50像素区域）
    """
    all_tables = []
    pending_tables = deque()  # 待合并表格缓存

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 获取页面元素信息
            text_elements = page.extract_words(keep_blank_chars=True)
            tables = page.find_tables()

            # 按表格从下到上处理（避免标题区域重叠）
            for table in reversed(tables):
                # 提取表格数据
                table_data = table.extract()
                if len(table_data) < 2:
                    continue  # 忽略空表

                # 转换为DataFrame
                df = pd.DataFrame(table_data[1:], columns=table_data[0])

                # 精准定位标题（基于坐标系统）
                table_bbox = table.bbox
                title_rect = (
                    table_bbox[0],              # x0
                    table_bbox[1] - title_search_height,  # y0
                    table_bbox[2],              # x1
                    table_bbox[1]               # y1
                )

                # 提取标题区域文本
                title = " ".join([
                    elem["text"] for elem in text_elements
                    if (elem["x0"] >= title_rect[0] and 
                        elem["x1"] <= title_rect[2] and
                        elem["top"] >= title_rect[1] and
                        elem["bottom"] <= title_rect[3])
                ]).strip() or "Untitled"

                # 跨页表格合并判断
                if pending_tables:
                    last_df = pending_tables[-1]["df"]
                    # 列名相似度检查（容错处理）
                    col_similarity = sum(a == b for a, b in zip(df.columns, last_df.columns)) / len(df.columns)
                    
                    if col_similarity > 0.8:  # 80%列名匹配即视为同一表格
                        merged_df = pd.concat([last_df, df], ignore_index=True)
                        pending_tables[-1] = {"df": merged_df, "title": pending_tables[-1]["title"]}
                        continue

                # 缓存新表格
                pending_tables.append({"df": df, "title": title})

            # 处理本页可确认的独立表格
            while len(pending_tables) > 1:
                confirmed_table = pending_tables.popleft()
                all_tables.append({
                    "title": confirmed_table["title"],
                    "table": confirmed_table["df"]
                })

        # 处理剩余表格
        all_tables.extend([
            {"title": t["title"], "table": t["df"]} 
            for t in pending_tables
        ])

    return all_tables

def extract_table_by_camelot(pdf_path):
    """
    1.由于 camelot 只提取表格，这里用 PDF 文本解析方法获取表格前面的文字，并自动匹配标题。
    2.camelot 解析 PDF 后，跨页的表格会被拆分，但它们的列名通常相同，可以作为合并的依据
    """

    all_tables = []
    last_table = None
    last_title = None

    # 获取 PDF 总页数
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

    for page in range(1, total_pages + 1):
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[page - 1].extract_text()

        tables = camelot.read_pdf(pdf_path, pages=str(page), flavor='stream')
        for table in tables:
            df = table.df

            # 提取标题
            lines = text.split("\n")
            first_row_text = " ".join(df.iloc[0].tolist())  # 获取表格第一行的文本
            table_start_idx = next((idx for idx, line in enumerate(lines) if first_row_text in line), None)
            title = "\n".join(lines[max(0, table_start_idx - 2): table_start_idx]) if table_start_idx else "Unknown Title"

            # 处理跨页表格合并
            if last_table is not None and list(df.columns) == list(last_table.columns):  
                last_table = pd.concat([last_table, df.iloc[1:]], ignore_index=True)  # 仅合并数据部分
            else:
                if last_table is not None:
                    all_tables.append({"title": last_title, "table": last_table})  # 保存上一张表
                last_table = df
                last_title = title  # 更新标题

    # 处理最后一张表格
    if last_table is not None:
        all_tables.append({"title": last_title, "table": last_table})

    return all_tables


if __name__ == '__main__':
    pdf_path = "../data/raw/fund/guotai.pdf"

    # dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    # tables = extract_table_by_pdfplumber(pdf_path)
    # import ipdb; ipdb.set_trace()
    # print("OK")

    tables = camelot.read_pdf(pdf_path, pages='all')
    import ipdb; ipdb.set_trace()
    # tables = []
    # with pdfplumber.open(pdf_path) as pdf:
    #     for page in pdf.pages:
    #         tables.extend(page.extract_tables())
    
    print("OK")

    # tables_with_titles = extract_table_with_title_and_merge(pdf_path)

    # for item in tables_with_titles:
    #     print("Title:", item["title"])
    #     print("\n")
    #     print("Table:", item["table"])
    #     print("\n ******************************** \n")
