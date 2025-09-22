import sys
from bs4 import BeautifulSoup

def extract_main_content(file_path, output_path):
    """
    从 HTML 文件中提取 class="VPDoc" 的 div 内容，移除样式后保存到输出文件。

    :param file_path: 输入的 HTML 文件路径。
    :param output_path: 输出文件路径。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 查找 class="VPDoc" 的 div
        vpdoc_div = soup.find('div', class_='VPDoc')

        if vpdoc_div:
            # 移除所有主要用于样式的 <span> 标签，但保留其内容
            for span in vpdoc_div.find_all('span'):
                span.unwrap()

            # 移除所有标签中的 style 属性
            for tag in vpdoc_div.find_all(True):
                if 'style' in tag.attrs:
                    del tag['style']

            # 获取清理后的 div 的内部 HTML 内容
            cleaned_html = vpdoc_div.prettify()
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(cleaned_html)
            print(f"Content extracted and cleaned to {output_path}")
        else:
            print("Error: <div class=\"VPDoc\"> not found.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_content.py <input_html_file> <output_html_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_main_content(input_file, output_file)