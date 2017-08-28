import bleach
import markdown

from bleach_whitelist.bleach_whitelist import markdown_tags, markdown_attrs


def format_doc(doc):
    doc = markdown.markdown(doc)
    doc = bleach.clean(doc,
                       tags=markdown_tags,
                       attributes=markdown_attrs,
                       )
    return doc
