import sys
import re
from bs4 import BeautifulSoup

def html_to_markdown(html_file, md_file):
    """
    将HTML文件转换为Markdown格式，保持原始内容不做优化或编造
    """
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 查找主要内容区域
        main_content = soup.find('div', class_='vp-doc')
        if not main_content:
            main_content = soup.find('div', class_='VPDoc')
        if not main_content:
            main_content = soup
        
        markdown_content = ""
        
        # 处理内容
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'img', 'table', 'ul', 'ol', 'blockquote']):
            if element.name.startswith('h'):
                # 处理标题
                level = int(element.name[1])
                title_text = element.get_text().strip()
                # 移除锚点符号
                title_text = re.sub(r'\s*​\s*$', '', title_text)
                markdown_content += '#' * level + ' ' + title_text + '\n\n'
            
            elif element.name == 'p':
                # 处理段落
                p_text = element.get_text().strip()
                if p_text:
                    markdown_content += p_text + '\n\n'
            
            elif element.name == 'pre':
                # 处理代码块
                code_element = element.find('code')
                if code_element:
                    # 获取语言类型
                    lang = ''
                    if code_element.get('class'):
                        for cls in code_element.get('class'):
                            if cls.startswith('language-'):
                                lang = cls.replace('language-', '')
                                break
                    
                    code_text = code_element.get_text()
                    markdown_content += f'```{lang}\n{code_text}\n```\n\n'
            
            elif element.name == 'img':
                # 处理图片
                src = element.get('src', '')
                alt = element.get('alt', '')
                # 更新图片路径
                if src.startswith('/assets/'):
                    src = src.replace('/assets/', './images/')
                markdown_content += f'![{alt}]({src})\n\n'
            
            elif element.name == 'table':
                # 处理表格
                markdown_content += convert_table_to_markdown(element) + '\n\n'
            
            elif element.name in ['ul', 'ol']:
                # 处理列表
                markdown_content += convert_list_to_markdown(element) + '\n\n'
            
            elif element.name == 'blockquote':
                # 处理引用
                quote_text = element.get_text().strip()
                lines = quote_text.split('\n')
                for line in lines:
                    if line.strip():
                        markdown_content += '> ' + line.strip() + '\n'
                markdown_content += '\n'
        
        # 写入Markdown文件
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Successfully converted {html_file} to {md_file}")
        
    except Exception as e:
        print(f"Error converting {html_file}: {e}")

def convert_table_to_markdown(table):
    """将HTML表格转换为Markdown格式"""
    markdown = ""
    rows = table.find_all('tr')
    
    for i, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        row_content = '| ' + ' | '.join(cell.get_text().strip() for cell in cells) + ' |'
        markdown += row_content + '\n'
        
        # 添加表头分隔符
        if i == 0 and row.find_all('th'):
            separator = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
            markdown += separator + '\n'
    
    return markdown

def convert_list_to_markdown(list_element):
    """将HTML列表转换为Markdown格式"""
    markdown = ""
    items = list_element.find_all('li', recursive=False)
    
    for i, item in enumerate(items):
        item_text = item.get_text().strip()
        if list_element.name == 'ol':
            markdown += f"{i+1}. {item_text}\n"
        else:
            markdown += f"- {item_text}\n"
    
    return markdown

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python html_to_md.py <input_html_file> <output_md_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    html_to_markdown(input_file, output_file)