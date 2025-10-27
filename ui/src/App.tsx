/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState } from "react";
import { queryApi } from "./api";

export default function App ()
{
  const [ text, setText ] = useState( "Which products drove revenue growth?" );
  const [ company, setCompany ] = useState( "AAPL" );
  const [ year, setYear ] = useState( 2023 );
  const [ ans, setAns ] = useState<string>( "" );
  const [ ctxCount, setCtxCount ] = useState<number>( 0 );
  const [ msg, setMsg ] = useState<string>( "" );

  const ask = async () =>
  {
    setMsg( "…" );
    try
    {
      const res = await queryApi( { text, company, year } );
      setAns( res.answer || "" );
      setCtxCount( ( res.contexts || [] ).length );
      setMsg( "" );
      if ( !res.contexts?.length ) setMsg( `No indexed filings for ${ company } ${ year }. Ingest first.` );
    } catch ( e: any )
    {
      setMsg( e.message || "Error" );
    }
  };

  return (
    <div style={ { maxWidth: 720, margin: "40px auto", fontFamily: "system-ui, sans-serif" } }>
      <h1>IntelSent Demo</h1>
      <div style={ { display: "grid", gap: 8 } }>
        <input value={ text } onChange={ e => setText( e.target.value ) } placeholder="Question" />
        <div style={ { display: "flex", gap: 8 } }>
          <input value={ company } onChange={ e => setCompany( e.target.value.toUpperCase() ) } style={ { width: 120 } } placeholder="Company" />
          <input type="number" value={ year } onChange={ e => setYear( parseInt( e.target.value || "0" ) ) } style={ { width: 120 } } placeholder="Year" />
          <button onClick={ ask }>Ask</button>
        </div>
      </div>
      { msg && <p style={ { color: "#a00" } }>{ msg }</p> }
      { ans && (
        <div style={ { marginTop: 12, padding: 12, border: "1px solid #ddd", borderRadius: 8 } }>
          <b>Answer</b>
          <p>{ ans }</p>
          <small>contexts: { ctxCount }</small>
        </div>
      ) }
    </div>
  );
}
