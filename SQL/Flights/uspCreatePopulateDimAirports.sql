
-- =============================================
-- Author:		Keshav Tyagi
-- Create date: Nov 23 2017
-- Description:	Creates a Dimension table for Airports
-- =============================================
CREATE PROCEDURE [dbo].[uspCreatePopulateDimAirports] 	
AS
BEGIN
BEGIN TRY
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	IF OBJECT_ID('[tmp].[DimAirports]','U') IS NOT NULL
	BEGIN
	select 1
	DROP TABLE [tmp].[DimAirports]
	END

    CREATE TABLE [tmp].[DimAirports](
	[AirportsKey] INT PRIMARY KEY IDENTITY,
	[IATA_CODE] [varchar](50) NULL,
	[AIRPORT] [varchar](255) NULL,
	[CITY] [varchar](50) NULL,
	[STATE] [varchar](50) NULL,
	[COUNTRY] [varchar](50) NULL,
	[LATITUDE] [varchar](50) NULL,
	[LONGITUDE] [varchar](50) NULL
) ON [PRIMARY]

	INSERT INTO [tmp].[DimAirports]
	SELECT DISTINCT [IATA_CODE]
      ,[AIRPORT]
      ,[CITY]
      ,[STATE]
      ,[COUNTRY]
      ,[LATITUDE]
      ,[LONGITUDE]
  FROM [UC].[dbo].[airports]

	 BEGIN TRANSACTION 
	 IF EXISTS( Select 1 FROM tmp.DimAirports)
	 BEGIN
	 IF OBJECT_ID('dbo.DimAirports','U') IS NOT NULL
	 BEGIN
	 DROP TABLE dbo.DimAirports
	 END
	 ALTER SCHEMA dbo TRANSFER tmp.DimAirports
	 END
	 COMMIT TRANSACTION 
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
