import { useState } from "react";

type QueryResponse = {
  answer: string;
  contexts: string[];
  meta?: Record<string, unknown>;
};

const API_BASE = import.meta.env.VITE_API_BASE as string;
const API_KEY = import.meta.env.VITE_API_KEY as string;

export default function App ()
{
  const [ text, setText ] = useState<string>( "Which products drove revenue growth?" );
  const [ company, setCompany ] = useState<string>( "AAPL" );
  const [ year, setYear ] = useState<string>( "2023" );

  const [ answer, setAnswer ] = useState<string>( "" );
  const [ contexts, setContexts ] = useState<string[]>( [] );
  const [ loading, setLoading ] = useState<boolean>( false );
  const [ err, setErr ] = useState<string>( "" );

  async function ask ( e: React.FormEvent )
  {
    e.preventDefault();
    setErr( "" );
    setAnswer( "" );
    setContexts( [] );
    setLoading( true );

    try
    {
      const qs = new URLSearchParams( {
        text,
        company: company.trim(),
        year: year.trim() ? String( Number( year.trim() ) ) : "",
        top_k: "3",
        no_openai: "true",
        api_key: API_KEY, // via query param to avoid preflight
      } );

      const res = await fetch( `${ API_BASE }/query_min?${ qs.toString() }`, { method: "GET" } );

      if ( !res.ok )
      {
        const msg = await res.text().catch( () => "" );
        throw new Error( `HTTP ${ res.status } ${ res.statusText } ${ msg || "" }`.trim() );
      }

      const data: QueryResponse = await res.json();
      setAnswer( data.answer || "" );
      setContexts( Array.isArray( data.contexts ) ? data.contexts : [] );
    } catch ( err: unknown )
    {
      const msg =
        err && typeof err === "object" && "message" in err
          ? String( ( err as { message?: string } ).message )
          : String( err );
      setErr( msg || "network error" );
    } finally
    {
      setLoading( false );
    }
  }

  return (
    <div
      style={ {
        fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial",
        padding: "28px",
        maxWidth: "980px",
        margin: "0 auto",
        lineHeight: 1.6,
        fontSize: "18px",
      } }
    >
      <h1 style={ { fontSize: "30px", marginBottom: "16px" } }>IntelSent — Demo UI v2</h1>

      <form onSubmit={ ask } style={ { display: "grid", gap: "12px", marginBottom: "16px" } }>
        <label style={ { display: "grid", gap: "6px" } }>
          <span>Question</span>
          <input
            value={ text }
            onChange={ ( e ) => setText( e.target.value ) }
            placeholder="Ask something…"
            disabled={ loading }
            style={ { padding: "12px", border: "1px solid #ddd", borderRadius: "8px" } }
          />
        </label>

        <div style={ { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" } }>
          <label style={ { display: "grid", gap: "6px" } }>
            <span>Company</span>
            <input
              value={ company }
              onChange={ ( e ) => setCompany( e.target.value.toUpperCase() ) }
              placeholder="AAPL"
              disabled={ loading }
              style={ { padding: "12px", border: "1px solid #ddd", borderRadius: "8px" } }
            />
          </label>
          <label style={ { display: "grid", gap: "6px" } }>
            <span>Year</span>
            <input
              value={ year }
              onChange={ ( e ) => setYear( e.target.value ) }
              placeholder="2023"
              inputMode="numeric"
              disabled={ loading }
              style={ { padding: "12px", border: "1px solid #ddd", borderRadius: "8px" } }
            />
          </label>
        </div>

        <button
          type="submit"
          disabled={ loading }
          style={ {
            padding: "12px 16px",
            borderRadius: "10px",
            border: "1px solid #111",
            background: loading ? "#f3f3f3" : "#111",
            color: loading ? "#111" : "#fff",
            fontWeight: 600,
            cursor: loading ? "default" : "pointer",
            marginTop: "4px",
            display: "inline-flex",
            alignItems: "center",
            gap: "10px",
          } }
          aria-busy={ loading }
          aria-live="polite"
        >
          { loading ? (
            <>
              {/* tiny inline SVG spinner */ }
              <svg width="18" height="18" viewBox="0 0 50 50" aria-hidden="true" role="img">
                <circle
                  cx="25"
                  cy="25"
                  r="20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="5"
                  strokeOpacity="0.25"
                />
                <path d="M25 5 a20 20 0 0 1 20 20" fill="none" stroke="currentColor" strokeWidth="5">
                  <animateTransform
                    attributeName="transform"
                    type="rotate"
                    from="0 25 25"
                    to="360 25 25"
                    dur="0.8s"
                    repeatCount="indefinite"
                  />
                </path>
              </svg>
              Asking…
            </>
          ) : (
            "Ask"
          ) }
        </button>
      </form>

      { err && (
        <div style={ { color: "#b00020", marginBottom: "12px", whiteSpace: "pre-wrap" } }>Error: { err }</div>
      ) }

      <div style={ { display: "grid", gap: "12px" } }>
        <div>
          <h2 style={ { fontSize: "22px", marginBottom: "8px" } }>Answer</h2>
          <div
            style={ {
              border: "1px solid #eee",
              borderRadius: "8px",
              padding: "12px",
              minHeight: "92px",
              background: "#fafafa",
              whiteSpace: "pre-wrap",
            } }
          >
            { answer || "—" }
          </div>
        </div>

        <div>
          <h2 style={ { fontSize: "22px", marginBottom: "8px" } }>Contexts ({ contexts.length })</h2>
          <ol style={ { display: "grid", gap: "8px", paddingLeft: "22px" } }>
            { contexts.map( ( c, i ) => (
              <li
                key={ i }
                style={ {
                  background: "#fcfcfc",
                  padding: "10px",
                  borderRadius: "8px",
                  border: "1px solid #eee",
                } }
              >
                <div style={ { whiteSpace: "pre-wrap" } }>{ c }</div>
              </li>
            ) ) }
            { !contexts.length && <div>—</div> }
          </ol>
        </div>
      </div>

      <div style={ { marginTop: "20px", color: "#666", fontSize: "14px" } }>
        API: { API_BASE || "(not set)" } · Key: { API_KEY ? `set (${ API_KEY.slice( 0, 8 ) }…)` : "missing" }
      </div>
    </div>
  );
}
