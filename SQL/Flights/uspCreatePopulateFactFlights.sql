

CREATE PROCEDURE [dbo].[usp_CreatePopulateFactFlights]
AS

BEGIN
BEGIN TRY

  IF OBJECT_ID('tmp.Factflights','U') IS NOT NULL
  DROP TABLE tmp.Factflights 

  CREATE TABLE [tmp].[Factflights](
	[datekey] [int] NULL,
	[AirlinesKey] [int] NOT NULL,
	[FLIGHT_NUMBER] [nvarchar](30) NULL,
	[TAIL_NUMBER] [varchar](30) NULL,
	[ORIGIN_AIRPORT_KEY] [varchar](50) NOT NULL,
	[DESTINATION_AIRPORT_KEY] [varchar](50) NOT NULL,
	[SCHEDULED_DEPARTURE] [nvarchar](30) NULL,
	[DEPARTURE_TIME] [nvarchar](30) NULL,
	[DEPARTURE_DELAY] [nvarchar](30) NULL,
	[TAXI_OUT] [nvarchar](30) NULL,
	[WHEELS_OFF] [nvarchar](30) NULL,
	[SCHEDULED_TIME] [nvarchar](30) NULL,
	[ELAPSED_TIME] [nvarchar](30) NULL,
	[AIR_TIME] [nvarchar](30) NULL,
	[DISTANCE] [nvarchar](30) NULL,
	[WHEELS_ON] [nvarchar](30) NULL,
	[TAXI_IN] [nvarchar](30) NULL,
	[SCHEDULED_ARRIVAL] [nvarchar](30) NULL,
	[ARRIVAL_TIME] [nvarchar](30) NULL,
	[ARRIVAL_DELAY] [nvarchar](30) NULL,
	[DIVERTED] [nvarchar](30) NULL,
	[CANCELLED] [nvarchar](30) NULL,
	[CANCELLATION_REASON] [varchar](30) NULL,
	[AIR_SYSTEM_DELAY] [nvarchar](30) NULL,
	[SECURITY_DELAY] [nvarchar](30) NULL,
	[AIRLINE_DELAY] [nvarchar](30) NULL,
	[LATE_AIRCRAFT_DELAY] [nvarchar](30) NULL,
	[WEATHER_DELAY] [nvarchar](30) NULL
) ON [PRIMARY]
  
  -- creating a refined table by excluding some null values
  INSERT INTO [tmp].[Factflights]
    SELECT  CONVERT(INT,SUBSTRING(YEAR,3,2)+right('0'+MONTH,2)+right('0'+DAY,2)) as [datekey]
		,ISNULL( DAL.[AirlinesKey],0) AS [AirlinesKey]
		,CONVERT( NVARCHAR, [FLIGHT_NUMBER]) AS [FLIGHT_NUMBER]
		,CONVERT( VARCHAR, [TAIL_NUMBER]) AS [TAIL_NUMBER]
		,ISNULL( DAP_origin.[AirportsKey],0) AS [ORIGIN_AIRPORT_KEY]
		,ISNULL( DAP_dest.[AirportsKey],0) AS [DESTINATION_AIRPORT_KEY]
		,CONVERT( NVARCHAR, [SCHEDULED_DEPARTURE]) AS [SCHEDULED_DEPARTURE]
		,CONVERT( NVARCHAR, [DEPARTURE_TIME]) AS [DEPARTURE_TIME]
		,CONVERT( NVARCHAR, [DEPARTURE_DELAY]) AS [DEPARTURE_DELAY]
		,CONVERT( NVARCHAR, [TAXI_OUT]) AS [TAXI_OUT]
		,CONVERT( NVARCHAR, [WHEELS_OFF]) AS [WHEELS_OFF]
		,CONVERT( NVARCHAR, [SCHEDULED_TIME]) AS [SCHEDULED_TIME]
		,CONVERT( NVARCHAR, [ELAPSED_TIME]) AS [ELAPSED_TIME]
		,CONVERT( NVARCHAR, [AIR_TIME]) AS [AIR_TIME]
		,CONVERT( NVARCHAR, [DISTANCE]) AS [DISTANCE]
		,CONVERT( NVARCHAR, [WHEELS_ON]) AS [WHEELS_ON]
		,CONVERT( NVARCHAR, [TAXI_IN]) AS [TAXI_IN]
		,CONVERT( NVARCHAR, [SCHEDULED_ARRIVAL]) AS [SCHEDULED_ARRIVAL]
		,CONVERT( NVARCHAR, [ARRIVAL_TIME]) AS [ARRIVAL_TIME]
		,CONVERT( NVARCHAR, [ARRIVAL_DELAY]) AS [ARRIVAL_DELAY]
		,CONVERT( NVARCHAR, [DIVERTED]) AS [DIVERTED]
		,CONVERT( NVARCHAR, [CANCELLED]) AS [CANCELLED]
		,CONVERT( VARCHAR, [CANCELLATION_REASON]) AS [CANCELLATION_REASON]
		,CONVERT( NVARCHAR, [AIR_SYSTEM_DELAY]) AS [AIR_SYSTEM_DELAY]
		,CONVERT( NVARCHAR, [SECURITY_DELAY]) AS [SECURITY_DELAY]
		,CONVERT( NVARCHAR, [AIRLINE_DELAY]) AS [AIRLINE_DELAY]
		,CONVERT( NVARCHAR, [LATE_AIRCRAFT_DELAY]) AS [LATE_AIRCRAFT_DELAY]
		,CONVERT( NVARCHAR, [WEATHER_DELAY]) AS [WEATHER_DELAY]
		FROM [UC].[dbo].[flights] F
		LEFT JOIN [dbo].[Dimairlines] DAL
		ON F.[Airline]= DAL.[IATA_CODE]
		LEFT JOIN [dbo].[DimAirports] DAP_origin
		ON DAP_origin.[IATA_CODE] = F.[ORIGIN_AIRPORT]
		LEFT JOIN [dbo].[DimAirports] DAP_dest
		ON  DAP_dest.[IATA_CODE] = F.[DESTINATION_AIRPORT]
		WHERE ISNULL(LTRIM(RTRIM(DEPARTURE_TIME)),'')<>'' AND 
		ISNULL(LTRIM(RTRIM(WHEELS_OFF)),'')<>'' 
		AND  ISNULL(LTRIM(RTRIM(AIR_TIME)),'')<>''
		 AND  ISNULL(LTRIM(RTRIM(TAXI_IN)),'') <>''
		 OR convert(int,diverted) = 1

		IF EXISTS (SELECT name FROM sys.indexes  
            WHERE name = N'NIX_FactFlights_datekey' AND object_id = (select object_id from sys.objects where name = 'Factflights' and schema_Name(schema_id) ='tmp'))   
		DROP INDEX NIX_FactFlights_datekey ON [tmp].[Factflights];   
		  
		-- Create a nonclustered index called NIX_FactFlights_datekey  
		-- on the dbo.flights_refined table using the datekey column.   
		CREATE NONCLUSTERED INDEX NIX_FactFlights_datekey
		    ON [tmp].[Factflights] (DateKey); 
			
		IF EXISTS (SELECT name  FROM sys.indexes  
            WHERE name = N'NIX_FactFlights_AirlinesKey' AND object_id = (select object_id from sys.objects where name = 'Factflights'and schema_Name(schema_id) ='tmp'))   
		DROP INDEX NIX_FactFlights_AirlinesKey ON [tmp].[Factflights];   
		  
		-- Create a nonclustered index called NIX_FactFlights_AirlinesKey  
		-- on the dbo.flights_refined table using the AirlinesKey column.   
		CREATE NONCLUSTERED INDEX NIX_FactFlights_AirlinesKey
		    ON [tmp].[Factflights] (AirlinesKey); 
			
	 BEGIN TRANSACTION 
	 IF EXISTS( Select 1 FROM [tmp].[Factflights])
	 BEGIN
	 IF OBJECT_ID('dbo.Factflights','U') IS NOT NULL
	 BEGIN
	 DROP TABLE [dbo].[Factflights]
	 END
	 ALTER SCHEMA dbo TRANSFER [tmp].[Factflights]
	 END
	 COMMIT TRANSACTION 

		--IF EXISTS (SELECT name FROM sys.indexes  
  --          WHERE name = N'NIX_FactFlights_datekey')   
		--DROP INDEX NIX_FactFlights_datekey ON dbo.flights_refined;   
		  
		---- Create a nonclustered index called NIX_FactFlights_datekey  
		---- on the dbo.flights_refined table using the datekey column.   
		--CREATE NONCLUSTERED INDEX NIX_FactFlights_datekey
		--    ON dbo.flights_refined (DateKey);   

 
END TRY

BEGIN CATCH 
  IF (@@TRANCOUNT > 0)
   BEGIN
      ROLLBACK TRANSACTION 
      PRINT 'Error detected, all changes reversed'
   END 
    SELECT
        ERROR_NUMBER() AS ErrorNumber,
        ERROR_SEVERITY() AS ErrorSeverity,
        ERROR_STATE() AS ErrorState,
        ERROR_PROCEDURE() AS ErrorProcedure,
        ERROR_LINE() AS ErrorLine,
        ERROR_MESSAGE() AS ErrorMessage
END CATCH
END