
-- =============================================
-- Author:		Keshav Tyagi
-- Create date: Nov 23 2017
-- Description:	Creates a Dimension table for Airline Carriers
-- =============================================
CREATE PROCEDURE [dbo].[uspCreatePopulateDimAirline] 	
AS
BEGIN
BEGIN TRY
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	IF OBJECT_ID('[tmp].[Dimairlines]','U') IS NOT NULL
	BEGIN
	select 1
	DROP TABLE [tmp].[Dimairlines]
	END

    CREATE TABLE [tmp].[Dimairlines](
	[AirlinesKey] INT PRIMARY KEY IDENTITY,
	[IATA_CODE] [varchar](50) NULL,
	[AIRLINE] [varchar](50) NULL
	) ON [PRIMARY]

	INSERT INTO [tmp].[Dimairlines]
	SELECT DISTINCT [IATA_CODE]
      ,[AIRLINE]
	FROM [UC].[dbo].[airlines]

	 BEGIN TRANSACTION 
	 IF EXISTS( Select 1 FROM tmp.Dimairlines)
	 BEGIN
	 IF OBJECT_ID('dbo.Dimairlines','U') IS NOT NULL
	 BEGIN
	 DROP TABLE dbo.Dimairlines
	 END
	 ALTER SCHEMA dbo TRANSFER tmp.Dimairlines
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
