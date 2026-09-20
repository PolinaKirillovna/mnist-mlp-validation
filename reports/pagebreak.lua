-- Pandoc filter: insert a Word page break before every level-1 heading.
-- Used by `make report` so page breaks live in the DOCX build, not in
-- report.md (which stays clean GitHub-flavoured markdown). report.md opens
-- with a centered raw-HTML title block (not a heading), so the first page
-- break falls before the first real section and the title stays on page 1.
function Header(el)
  if el.level == 1 then
    local pb = pandoc.RawBlock('openxml',
      '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    return { pb, el }
  end
end
