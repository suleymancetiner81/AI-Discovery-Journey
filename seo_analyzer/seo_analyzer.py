import requests
from bs4 import BeautifulSoup
import extruct
from w3lib.html import get_base_url
from rich import print
from rich.panel import Panel
from rich.table import Table

def fetch_html(url):
    try:
        headers = {"User-Agent": "SEOlyzerBot/1.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text, response.url
    except Exception as e:
        print(f"[red]Hata oluştu: {e}[/red]")
        return None, None

def analyze_metatags(soup):
    print()
    print(Panel("🎯 [bold cyan]Meta Tag Analizi[/bold cyan]", expand=False))

    title = soup.title.string.strip() if soup.title else None
    meta_desc = soup.find("meta", attrs={"name": "description"})
    robots = soup.find("meta", attrs={"name": "robots"})

    if title:
        print(f"[green]Title:[/green] {title} ({len(title)} karakter)")
    else:
        print("[red]Eksik:[/red] <title> etiketi bulunamadı.")

    if meta_desc:
        content = meta_desc.get("content", "")
        print(f"[green]Meta Description:[/green] {content} ({len(content)} karakter)")
    else:
        print("[red]Eksik:[/red] Meta description bulunamadı.")

    if robots:
        print(f"[green]Meta Robots:[/green] {robots.get('content')}")
    else:
        print("[yellow]Uyarı:[/yellow] robots etiketi yok (isteğe bağlı).")

def analyze_structured_data(html, url):
    print()
    print(Panel("🔎 [bold cyan]Structured Data (JSON-LD)[/bold cyan]", expand=False))

    data = extruct.extract(html, base_url=get_base_url(html, url), syntaxes=['json-ld'])
    jsonld = data.get("json-ld", [])

    if not jsonld:
        print("[red]JSON-LD structured data bulunamadı.[/red]")
    else:
        table = Table(title="Bulunan JSON-LD Nesneleri")
        table.add_column("Tip", style="cyan")
        table.add_column("Alanlar", style="green")

        for entry in jsonld:
            entry_type = entry.get("@type", "Bilinmiyor")
            fields = ", ".join(entry.keys())
            table.add_row(str(entry_type), fields)

        print(table)

def main():
    url = input("🔗 SEO analizi yapılacak URL: ").strip()
    html, final_url = fetch_html(url)
    if html:
        soup = BeautifulSoup(html, "lxml")
        analyze_metatags(soup)
        analyze_structured_data(html, final_url)

if __name__ == "__main__":
    main()